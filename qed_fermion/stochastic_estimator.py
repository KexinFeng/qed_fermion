import torch
import os 
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.util import ravel_multi_index, unravel_index

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')
from qed_fermion.hmc_sampler_batch import HmcSampler

if torch.cuda.is_available():
    from qed_fermion import _C 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = (4, 8)
print(f"BLOCK_SIZE: {BLOCK_SIZE}")

class StochaticEstimator:
    # This function computes the fermionic green's function, mainly four-point function. The aim is to surrogate the postprocessing of the HMC boson samples. For a given boson, the M(boson) is the determined, so is M_inv aka green's function.

    # So the algorithm is to generate Nrv iid random variables and utilize the orthogonaliaty of the random variables expectation, to estimate M_inv
    #   G_ij ~ (G eta)_i eta_j
    #   G_ij G_kl ~ (G eta)_i eta_j (G eta')_k eta'_l

    # So one method to generate Nrv eta vector: random_vec 
    # One method to estimate the four-point green's function: four_point_green
    # One method to compute the obs like spsm szsz, etc.

    def __init__(self, hmc_sampler):
        self.hmc_sampler = hmc_sampler
        self.Nrv = 10
        self.green_four = None
        self.green_two = None

        self.Lx = hmc_sampler.Lx
        self.Ly = hmc_sampler.Ly    
        self.Ltau = hmc_sampler.Ltau
        self.dtau = hmc_sampler.dtau

        self.Vs = hmc_sampler.Vs

        self.use_cuda_graph = hmc_sampler
        self.device = hmc_sampler.device

        # init
        self.hmc_sampler.reset_precon()
    
    def random_vec_bin(self):  
        """
        Generate Nrv iid random variables
    
        return: Nrv random complex vectors with entries from {1, -1, 1j, -1j} / sqrt(2), [Nrv, Lx * Ly * Ltau]
        """
        # Generate Nrv random complex vectors with entries from {1, -1, 1j, -1j} / sqrt(2)
        N_site = self.Lx * self.Ly * self.Ltau

        real_part = torch.randint(0, 2, (self.Nrv, N_site), dtype=torch.float32, device=device) * 2 - 1  # { -1, 1 }
        imag_part = torch.randint(0, 2, (self.Nrv, N_site), dtype=torch.float32, device=device) * 2 - 1  # { -1, 1 }

        rand_vec = (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(2.0, device=device))

        return rand_vec

    def random_vec_norm(self):
        """
        Generate Nrv iid real Gaussian random variables

        return: Nrv random real vectors with entries ~ N(0, 1/sqrt(2)), [Nrv, Lx * Ly * Ltau]
        """
        N_site = self.Lx * self.Ly * self.Ltau

        rand_vec = torch.randn(self.Nrv, N_site, device=device)
        return rand_vec

    @staticmethod
    def fft_negate_k(a_F):
        return torch.roll(a_F.flip(dims=[-1]), shifts=1, dims=-1)
    
    @staticmethod
    def fft_negate_k3(a_F):
        """
        Negate the momentum for 3D FFT output tensor a_F.
        Applies flip and roll on the last 3 dimensions.
        """
        # a_F: [..., T, Y, X]
        a_F = a_F.flip(dims=[-3, -2, -1])
        a_F = torch.roll(a_F, shifts=(1, 1, 1), dims=(-3, -2, -1))
        return a_F


    def test_orthogonality(self, rand_vec):
        """
        Test the orthogonality of the random vectors
        """
        # Compute the inner product of the random vectors
        external_product = torch.einsum('ai,aj->aij', rand_vec.conj(), rand_vec)
        external_product = external_product.mean(dim=0)

        print("Inner product shape:", external_product.shape)

        # Filter out entries of external_product that are smaller than 1e-3
        mask = external_product.abs() < 1e-3
        external_product[mask] = 0
        
        print("external_product.real[0, :10]:\n", external_product.real[0, :10])
        print("external_product.real[1, :10]:\n", external_product.real[1, :10])
        print("external_product.real[-1, -10:]:\n", external_product.real[-1, -10:])
    
        diff = external_product - torch.eye(external_product.size(0), device=device)  # [Nrv, Nrv]
        atol = diff.abs().max()
        print("Max absolute difference from orthogonality (atol):", atol.item())
        return atol

    def set_eta_G_eta(self, boson, eta):
        """
        Compute the four-point green's function

        boson: [bs=1, 2, Lx, Ly, Ltau]
        eta: [Nrv, Ltau * Ly * Lx]
        """
        # Compute the four-point green's function
        # G_ij ~ (G eta)_i eta_j
        # G_ij G_kl ~ (G eta)_i eta_j (G eta')_k eta'_l
        self.eta = eta  # [Nrv, Ltau * Ly * Lx]

        boson_in = boson.clone()

        boson = boson.permute([0, 4, 3, 2, 1]).reshape(1, -1).repeat(self.Nrv, 1)  # [Nrv, Ltau * Ly * Lx]

        psudo_fermion = _C.mh_vec(boson, eta, self.Lx, self.dtau, *BLOCK_SIZE)  # [Nrv, Ltau * Ly * Lx]

        self.hmc_sampler.bs = self.Nrv
        self.G_eta, cnt, err = self.hmc_sampler.Ot_inv_psi_fast(psudo_fermion, boson.view(self.Nrv, self.Ltau, -1), None)  # [Nrv, Ltau * Ly * Lx]
        self.hmc_sampler.bs = 1
        print("max_pcg_iter:", cnt[:10])
        print("err:", err[:10])

        # Check
        M, _ = self.hmc_sampler.get_M_batch(boson_in)
        M = M[0]
        psudo_fermion_ref = torch.einsum('ij,bj->bi', M.conj().T, eta)
        torch.testing.assert_close(psudo_fermion, psudo_fermion_ref, rtol=1e-3, atol=1e-3)

        O = M.conj().T @ M
        O_inv = torch.linalg.inv(O)
        
        G_eta_ref = torch.einsum('ij,bj->bi', O_inv, psudo_fermion_ref)
        torch.testing.assert_close(self.G_eta, G_eta_ref, rtol=1e-3, atol=1e-3)

        dbstop = 1
  
    def test_fft_negate_k3(self):
        device = self.device
        # Create 3D frequency grid for (2*Ltau, Ly, Lx)
        k_tau = torch.fft.fftfreq(2 * self.Ltau, device=device)
        k_y = torch.fft.fftfreq(self.Ly, device=device)
        k_x = torch.fft.fftfreq(self.Lx, device=device)
        ks = torch.stack(torch.meshgrid(k_tau, k_y, k_x, indexing='ij'), dim=-1)  # shape: (2*Ltau, Ly, Lx, 3)
        ks_neg = self.fft_negate_k3(ks.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        # dbstop = 1
        ktau_neg = self.fft_negate_k(k_tau)
        ky_neg = self.fft_negate_k(k_y)
        kx_neg = self.fft_negate_k(k_x)
        ks_neg_ref = torch.stack(torch.meshgrid(ktau_neg, ky_neg, kx_neg, indexing='ij'), dim=-1)  # shape: (2*Ltau, Ly, Lx, 3)
        torch.testing.assert_close(ks_neg, ks_neg_ref, rtol=1e-2, atol=1e-2)

    def G_delta_0(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        # Build augmented eta and G_eta
        # eta_ext = [eta, -eta], G_eta_ext = [G_eta, -G_eta]
        eta_ext_conj = eta.conj()
        G_eta_ext = G_eta

        # Compute the two-point green's function
        # Here, a = eta_ext, b = G_eta_ext
        a = eta_ext_conj.view(self.Nrv, self.Ltau, self.Ly, self.Lx)  # [Nrv, Ltau, Ly, Lx]
        b = G_eta_ext.view(self.Nrv, self.Ltau, self.Ly, self.Lx)  # [Nrv, Ltau, Ly, Lx]

        a_F_neg_k = torch.fft.ifftn(a, (self.Ltau, self.Ly, self.Lx), norm="backward")

        b_F = torch.fft.fftn(b, (self.Ltau, self.Ly, self.Lx), norm="forward")

        G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)   # [Ltau, Ly, Lx]
        return G_delta_0.permute(2, 1, 0) # [Lx, Ly, Ltau]

    def G_delta_0_ext(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        # eta_ext = [eta, -eta], G_eta_ext = [G_eta, -G_eta]
        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj()
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Compute the two-point green's function
        # Here, a = eta_ext, b = G_eta_ext
        a = eta_ext_conj.view(self.Nrv, 2*self.Ltau, self.Ly, self.Lx)  # [Nrv, 2Ltau, Ly, Lx]
        b = G_eta_ext.view(self.Nrv, 2*self.Ltau, self.Ly, self.Lx)  # [Nrv, 2Ltau, Ly, Lx]

        a_F = torch.fft.fftn(a, (2*self.Ltau, self.Ly, self.Lx), norm="forward")
        a_F_neg_k = self.fft_negate_k3(a_F)

        b_F = torch.fft.fftn(b, (2*self.Ltau, self.Ly, self.Lx), norm="forward")

        G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (2*self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)  # [2Ltau, Ly, Lx]
        return G_delta_0[:self.Ltau].permute(2, 1, 0) # [Lx, Ly, Ltau]

    def G_delta_0_O2(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]
        eta_conj = eta.conj()
        G_eta = G_eta

        G = torch.einsum('bi,bj->ji', eta_conj, G_eta) / self.Nrv  # [Ltau * Ly * Lx]
        
        N = G.shape[0]
        Ltau, Ly, Lx = self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )
                result[i, d] = G[idx, i]
        G_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return G_mean.view(self.Ltau, self.Ly, self.Lx).permute(2, 1, 0)  # [Lx, Ly, Ltau]

    def G_delta_0_O2_ext(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        # eta_ext = [eta, -eta], G_eta_ext = [G_eta, -G_eta]
        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj()
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        G = torch.einsum('bi,bj->ji', eta_ext_conj, G_eta_ext) / self.Nrv  # [2Ltau * Ly * Lx]
        
        Ltau2 = 2 * self.Ltau
        Lx = self.Lx
        Ly = self.Ly

        N = G.shape[0]
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=device), (Ltau2, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=device), (Ltau2, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau2, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau2, Ly, Lx)
                )
                result[i, d] = G[idx, i]
                
        G_mean = result.mean(dim=0)  # [2Ltau * Ly * Lx]
        return G_mean.view(Ltau2, Ly, Lx)[:self.Ltau].permute(2, 1, 0)  # [Lx, Ly, Ltau]
        
    def G_delta_0_G_delta_0(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        # Build augmented eta and G_eta
        # eta_ext = [eta, -eta], G_eta_ext = [G_eta, -G_eta]
        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj()
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Compute the four-point green's function
        # G_delta_0_G_delta_0
        # Get all unique pairs (s, s_prime) with s < s_prime
        N = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(N, N, offset=1)

        a = eta_ext_conj[s] * eta_ext_conj[s_prime]
        b = G_eta_ext[s] * G_eta_ext[s_prime]

        a_F = torch.fft.fft(a, dim=1, norm="forward")
        # To get a_F(-k), use torch.flip on the frequency dimension
        a_F_neg_k = self.fft_negate_k(a_F)

        # ks = torch.fft.fftfreq(a_F.shape[1]).view(1, -1)  # [Ltau * Ly * Lx]
        # ks_neg = self.fft_negate_k(ks)
        # dbstop = 1

        b_F = torch.fft.fft(b, dim=1, norm="forward")
        G_delta_0_G_delta_0 = torch.fft.ifft(a_F_neg_k * b_F, dim=1, norm="forward").mean(dim=(0)).view(2*self.Ltau, self.Ly, self.Lx)

        return G_delta_0_G_delta_0[:self.Ltau].permute(2, 1, 0)  # [Lx, Ly, Ltau]

    def G_delta_0_G_delta_0_O2(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        # Compute the four-point Green's function estimator
        # G_ijkl = <eta_i^* G_eta_j eta_k^* G_eta_l>
        # We want mean_i G_{i+d, i} * G_{i+d, i}
        G = torch.einsum('bi,bj->ji', eta.conj(), G_eta) / self.Nrv  # [N, N], N = Ltau * Ly * Lx

        N = G.shape[0]
        Ltau, Ly, Lx = self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[idx, i]
        GG_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return GG_mean.view(self.Ltau, self.Ly, self.Lx).permute(2, 1, 0)  # [Lx, Ly, Ltau]

    def G_groundtruth(self, boson):
        M, _ = self.hmc_sampler.get_M_batch(boson)
        M_inv = torch.linalg.inv(M[0])
        return M_inv  # [Ltau * Ly * Lx, Ltau * Ly * Lx]
    
    def G_delta_0_groundtruth(self, M_inv):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[idx, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i]
        """
        G = M_inv  # [N, N]: N = Ltau * Ly * Lx
        N = G.shape[0]
        Ltau, Ly, Lx = self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )
                result[i, d] = G[idx, i]
        G_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return G_mean.view(self.Ltau, self.Ly, self.Lx).permute(2, 1, 0)  # [Lx, Ly, Ltau]
    
    def G_delta_0_groundtruth_ext(self, M_inv):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[idx, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i]
        """
        G = M_inv  # [N, N]: N = Ltau * Ly * Lx
        
        Ltau2 = 2 * self.Ltau
        Lx = self.Lx
        Ly = self.Ly

        # Block concat: [[G, -G], [-G, G]] for G of shape [N, N]
        G = torch.cat([
            torch.cat([G, -G], dim=1),
            torch.cat([-G, G], dim=1)
        ], dim=0)  # [2N, 2N]

        N = G.shape[0]
        Ltau, Ly, Lx = 2*self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] 
        G_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return G_mean.view(Ltau2, Ly, Lx)[:self.Ltau].permute(2, 1, 0)  # [Lx, Ly, Ltau]

    def G_delta_0_G_delta_0_groundtruth(self, M_inv):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[i+d, i] * G[i+d, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i] * G[(i+d)%N, i]
        """

        # Compute the four-point green's function using the ground truth
        # G_delta_0_G_delta_0 = mean_i G_{i+d, i} G_{i+d, i}
        G = M_inv # [N, N]
        N = G.shape[0]
        Ltau, Ly, Lx = self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[idx, i]

        GG = result.mean(dim=0) # [Ltau * Ly * Lx]
        return GG.view(self.Ltau, self.Ly, self.Lx).permute(2, 1, 0)  # [Lx, Ly, Ltau]


if __name__ == "__main__":  
    # Set random seed for reproducibility
    torch.manual_seed(42)

    hmc = HmcSampler()
    hmc.Lx = 2
    hmc.Ly = 2
    hmc.Ltau = 2

    hmc.bs = 1
    hmc.reset()

    se = StochaticEstimator(hmc)
    se.Nrv = 100_000  # bs > 10000 will fail on _C.mh_vec, due to grid = {Ltau, bs}.
    se.Nrv = 10  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent
    
    se.test_orthogonality(se.random_vec_bin())
    # se.test_orthogonality(se.random_vec_norm())

    se.test_fft_negate_k3()

    # Compute Green prepare
    eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]
    # eta = se.random_vec_norm().to(torch.complex64)  # [Nrv, Ltau * Ly * Lx]
    boson = hmc.boson

    se.set_eta_G_eta(boson, eta)
    Gij_gt = se.G_groundtruth(boson)

    # Test Green
    G_stoch = se.G_delta_0()
    G_stoch_O2 = se.G_delta_0_O2()
    torch.testing.assert_close(G_stoch.real, G_stoch_O2.real, rtol=1e-2, atol=1e-2)
    
    G_gt = se.G_delta_0_groundtruth(Gij_gt)
    torch.testing.assert_close(G_gt.real, G_stoch_O2.real, rtol=1e-2, atol=2e-2)






    # # Test Green extended
    # G_stoch_ext = se.G_delta_0_ext()
    # G_stoch_O2_ext = se.G_delta_0_O2_ext()
    
    # G_gt_ext = se.G_delta_0_groundtruth_ext(Gij_gt)
    # torch.testing.assert_close(G_gt_ext.real, G_stoch_O2_ext.real, rtol=1e-2, atol=1e-2)

    # torch.testing.assert_close(G_stoch_ext.real, G_stoch_O2_ext.real, rtol=1e-2, atol=1e-2)

    # dbstop = 1

    # # Test Green four-point
    # GG_stoch_O2 = se.G_delta_0_G_delta_0_O2()
    # GG_stoch = se.G_delta_0_G_delta_0()

    # dbstop = 1

    # # Benchmark with direct inverse of M(boson)
    # G = se.G_groundtruth(boson)
    # GG_gt = se.G_delta_0_G_delta_0_groundtruth(G)
    # G_gt = se.G_delta_0_groundtruth(G)

    # # torch.testing.assert_close(G1.real, G1_gt.real, rtol=1e-2, atol=1e-2)

    # GG_gt = GG_gt.real
    # GG_gt[GG_gt.abs() < 1e-3] = 0

    # G_gt = G_gt.real
    # G_gt[G_gt.abs() < 1e-3] = 0

    # G1 = G_stoch_O2.real
    # G1[G1.abs() < 1e-3] = 0
    # torch.testing.assert_close(G1, G_gt, rtol=1e-2, atol=2e-2)

    # GG1 = GG_stoch_O2.real
    # GG1[GG1.abs() < 1e-3] = 0
    # torch.testing.assert_close(GG1, GG_gt, rtol=1e-2, atol=2e-2)

    # dbstop = 1
