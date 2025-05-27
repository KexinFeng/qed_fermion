import numpy as np
import torch
import os 
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.fermion_obsr_graph_runner import FermionObsrGraphRunner
from qed_fermion.utils.util import ravel_multi_index, unravel_index

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')
from qed_fermion.hmc_sampler_batch import HmcSampler

if torch.cuda.is_available():
    from qed_fermion import _C 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, hmc):
        self.hmc_sampler = hmc
        self.Nrv = 10
        self.green_four = None
        self.green_two = None

        self.Lx = hmc.Lx
        self.Ly = hmc.Ly    
        self.Ltau = hmc.Ltau
        self.dtau = hmc.dtau

        self.Vs = hmc.Vs

        self.cuda_graph = hmc.cuda_graph
        self.device = hmc.device
        self.dtype = hmc.dtype
        self.cdtype = hmc.cdtype
        
        self.graph_runner = FermionObsrGraphRunner(self)
        self.graph_memory_pool = None
        self.max_iter = 500

        # init
        self.hmc_sampler.reset_precon()

    def init_cuda_graph(self):
        hmc = self.hmc_sampler
        # Capture
        max_iter = self.max_iter
        if self.cuda_graph:
            print("Initializing CUDA graph for get_fermion_obsr...")
            dummy_eta = torch.zeros((self.Nrv, self.Ltau * self.Vs), device=hmc.device, dtype=hmc.cdtype)
            dummy_bosons = torch.zeros((hmc.bs, 2, self.Lx, self.Ly, self.Ltau), device=hmc.device, dtype=hmc.dtype)
            self.graph_memory_pool = self.graph_runner.capture(
                                        dummy_bosons, 
                                        dummy_eta, 
                                        max_iter = max_iter,
                                        graph_memory_pool=self.graph_memory_pool)
            print(f"CUDA graph initialization complete")

    def random_vec_bin(self):  
        """
        Generate Nrv iid random variables
    
        return: Nrv random complex vectors with entries from {1, -1, 1j, -1j} / sqrt(2), [Nrv, Lx * Ly * Ltau]
        """
        # Generate Nrv random complex vectors with entries from {1, -1, 1j, -1j} / sqrt(2)
        N_site = self.Lx * self.Ly * self.Ltau

        real_part = torch.randint(0, 2, (self.Nrv, N_site), dtype=torch.float32, device=self.device) * 2 - 1  # { -1, 1 }
        imag_part = torch.randint(0, 2, (self.Nrv, N_site), dtype=torch.float32, device=self.device) * 2 - 1  # { -1, 1 }

        rand_vec = (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(2.0, device=self.device))

        return rand_vec

    def random_vec_norm(self):
        """
        Generate Nrv iid real Gaussian random variables

        return: Nrv random real vectors with entries ~ N(0, 1/sqrt(2)), [Nrv, Lx * Ly * Ltau]
        """
        N_site = self.Lx * self.Ly * self.Ltau

        rand_vec = torch.randn(self.Nrv, N_site, device=self.device)
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

    @staticmethod
    def reorder_fft_grid2(tensor2d):
        """Reorder the last two axes of a tensor from FFT-style to ascending momentum order."""
        Ny, Nx = tensor2d.shape[-2], tensor2d.shape[-1]
        return torch.roll(tensor2d, shifts=(Ny // 2, Nx // 2), dims=(-2, -1))


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
    
        diff = external_product - torch.eye(external_product.size(0), device=self.device)  # [Nrv, Nrv]
        atol = diff.abs().max()
        print("Max absolute difference from orthogonality (atol):", atol.item())
        return atol

    def set_eta_G_eta_debug(self, boson, eta):
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

        self.hmc_sampler.bs, bs = self.Nrv, self.hmc_sampler.bs
        self.G_eta, cnt, err = self.hmc_sampler.Ot_inv_psi_fast(psudo_fermion, boson.view(self.Nrv, self.Ltau, -1), None)  # [Nrv, Ltau * Ly * Lx]
        self.hmc_sampler.bs = bs
        print("max_pcg_iter:", cnt[:5])
        print("err:", err[:5])

        # Check
        M, _ = self.hmc_sampler.get_M_batch(boson_in)
        M = M[0]
        psudo_fermion_ref = torch.einsum('ij,bj->bi', M.conj().T, eta)
        torch.testing.assert_close(psudo_fermion, psudo_fermion_ref, rtol=1e-3, atol=1e-3)

        O = M.conj().T @ M
        O_inv = torch.linalg.inv(O)
        
        G_eta_ref = torch.einsum('ij,bj->bi', O_inv, psudo_fermion_ref)
        torch.testing.assert_close(self.G_eta, G_eta_ref, rtol=1e-3, atol=1e-3)

  
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

        boson = boson.permute([0, 4, 3, 2, 1]).reshape(1, -1).repeat(self.Nrv, 1)  # [Nrv, Ltau * Ly * Lx]

        psudo_fermion = _C.mh_vec(boson, eta, self.Lx, self.dtau, *BLOCK_SIZE)  # [Nrv, Ltau * Ly * Lx]

        self.hmc_sampler.bs, bs = self.Nrv, self.hmc_sampler.bs
        self.G_eta, cnt, err = self.hmc_sampler.Ot_inv_psi_fast(psudo_fermion, boson.view(self.Nrv, self.Ltau, -1), None)  # [Nrv, Ltau * Ly * Lx]
        self.hmc_sampler.bs = bs

        # print("max_pcg_iter:", cnt[:5])
        # print("err:", err[:5])

  
    def test_fft_negate_k3(self):
        device = self.device
        # Create 3D frequency grid for (2*Ltau, Ly, Lx)
        k_tau = torch.fft.fftfreq(2 * self.Ltau, device=self.device)
        k_y = torch.fft.fftfreq(self.Ly, device=self.device)
        k_x = torch.fft.fftfreq(self.Lx, device=self.device)
        ks = torch.stack(torch.meshgrid(k_tau, k_y, k_x, indexing='ij'), dim=-1)  # shape: (2*Ltau, Ly, Lx, 3)
        ks_neg = self.fft_negate_k3(ks.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        # dbstop = 1
        ktau_neg = self.fft_negate_k(k_tau)
        ky_neg = self.fft_negate_k(k_y)
        kx_neg = self.fft_negate_k(k_x)
        ks_neg_ref = torch.stack(torch.meshgrid(ktau_neg, ky_neg, kx_neg, indexing='ij'), dim=-1)  # shape: (2*Ltau, Ly, Lx, 3)
        torch.testing.assert_close(ks_neg, ks_neg_ref, rtol=1e-2, atol=5e-2)

    # -------- FFT methods --------
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
        return G_delta_0

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

        a_F_neg_k = torch.fft.ifftn(a, (2*self.Ltau, self.Ly, self.Lx), norm="backward")
        b_F = torch.fft.fftn(b, (2*self.Ltau, self.Ly, self.Lx), norm="forward")

        G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (2*self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)  # [2Ltau, Ly, Lx]
        return G_delta_0[:self.Ltau]

    def G_delta_0_G_delta_0(self):
        eta_conj = self.eta.conj()  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        # Get all unique pairs (s, s_prime) with s < s_prime
        N = eta_conj.shape[0]
        s, s_prime = torch.triu_indices(N, N, offset=1, device=self.eta.device)
        a = eta_conj[s] * eta_conj[s_prime]
        b = G_eta[s] * G_eta[s_prime]

        a = a.view(-1, self.Ltau, self.Ly, self.Lx)  # [N, Ltau, Ly, Lx]
        b = b.view(-1, self.Ltau, self.Ly, self.Lx)  # [N, Ltau, Ly, Lx]

        a_F_neg_k = torch.fft.ifftn(a, (self.Ltau, self.Ly, self.Lx), norm="backward")
        b_F = torch.fft.fftn(b, (self.Ltau, self.Ly, self.Lx), norm="forward")
        G_delta_0_G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F,  (self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)

        return G_delta_0_G_delta_0.view(self.Ltau, self.Ly, self.Lx)


    def G_delta_0_G_delta_0_ext(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj()
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Get all unique pairs (s, s_prime) with s < s_prime
        N = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(N, N, offset=1, device=eta.device)
        a = eta_ext_conj[s] * eta_ext_conj[s_prime]
        b = G_eta_ext[s] * G_eta_ext[s_prime]

        a = a.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]
        b = b.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]

        a_F_neg_k = torch.fft.ifftn(a, (2*self.Ltau, self.Ly, self.Lx), norm="backward")
        b_F = torch.fft.fftn(b, (2*self.Ltau, self.Ly, self.Lx), norm="forward")
        G_delta_0_G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (2*self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)

        return G_delta_0_G_delta_0.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]


    def G_delta_delta_G_0_0_ext(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj()
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Get all unique pairs (s, s_prime) with s < s_prime
        N = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(N, N, offset=1, device=eta.device)

        a = eta_ext_conj[s] * G_eta_ext[s]
        b = eta_ext_conj[s_prime] * G_eta_ext[s_prime]

        a = a.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]
        b = b.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]

        a_F_neg_k = torch.fft.ifftn(a, (2*self.Ltau, self.Ly, self.Lx), norm="backward")
        b_F = torch.fft.fftn(b, (2*self.Ltau, self.Ly, self.Lx), norm="forward")
        G_delta_0_G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (2*self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)

        return G_delta_0_G_delta_0.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]


    def G_delta_0_G_0_delta_ext(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj()
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Get all unique pairs (s, s_prime) with s < s_prime
        N = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(N, N, offset=1, device=eta.device)

        a = eta_ext_conj[s] * G_eta_ext[s_prime]
        b = eta_ext_conj[s_prime] * G_eta_ext[s]

        a = a.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]
        b = b.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]

        a_F_neg_k = torch.fft.ifftn(a, (2*self.Ltau, self.Ly, self.Lx), norm="backward")
        b_F = torch.fft.fftn(b, (2*self.Ltau, self.Ly, self.Lx), norm="forward")
        G_delta_0_G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (2*self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)

        return G_delta_0_G_delta_0.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]


    # -------- Primitive methods --------
    def G_delta_0_primitive(self):
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
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )
                result[i, d] = G[idx, i]
        G_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return G_mean.view(self.Ltau, self.Ly, self.Lx)


    def G_delta_0_primitive_ext(self):
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
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau2, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau2, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau2, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau2, Ly, Lx)
                )
                result[i, d] = G[idx, i]
                
        G_mean = result.mean(dim=0)  # [2Ltau * Ly * Lx]
        return G_mean.view(Ltau2, Ly, Lx)[:self.Ltau]


    def G_delta_0_G_delta_0_primitive(self):
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
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[idx, i]
        GG_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return GG_mean.view(self.Ltau, self.Ly, self.Lx)


    def G_delta_0_G_delta_0_primitive_ext(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext = torch.cat([eta, -eta], dim=1)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Compute the four-point Green's function estimator
        # G_ijkl = <eta_i^* G_eta_j eta_k^* G_eta_l>
        # We want mean_i G_{i+d, i} * G_{i+d, i}
        G = torch.einsum('bi,bj->ji', eta_ext.conj(), G_eta_ext) / self.Nrv  # [N, N], N = Ltau * Ly * Lx

        N = G.shape[0]
        Ltau, Ly, Lx = 2*self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[idx, i]
        GG_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return GG_mean.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]


    # -------- Ground truth methods --------
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
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )
                result[i, d] = G[idx, i]
        G_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return G_mean.view(self.Ltau, self.Ly, self.Lx)

    
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
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] 
        G_mean = result.mean(dim=0)  # [Ltau * Ly * Lx]
        return G_mean.view(Ltau2, Ly, Lx)[:self.Ltau]


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
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[idx, i]

        GG = result.mean(dim=0) # [Ltau * Ly * Lx]
        return GG.view(self.Ltau, self.Ly, self.Lx)


    def G_delta_0_G_delta_0_groundtruth_ext(self, M_inv):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[i+d, i] * G[i+d, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i] * G[(i+d)%N, i]
        """

        # Compute the four-point green's function using the ground truth
        # G_delta_0_G_delta_0 = mean_i G_{i+d, i} G_{i+d, i}
        G = M_inv # [N, N]

        # Block concat: [[G, -G], [-G, G]] for G of shape [N, N]
        G = torch.cat([
            torch.cat([G, -G], dim=1),
            torch.cat([-G, G], dim=1)
        ], dim=0)  # [2N, 2N]

        N = G.shape[0]
        Ltau, Ly, Lx = 2 * self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[idx, i]

        GG = result.mean(dim=0) # [Ltau * Ly * Lx]
        return GG.view(Ltau, Ly, Lx)[:self.Ltau]


    def G_delta_delta_G_0_0_groundtruth_ext(self, M_inv):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[i+d, i] * G[i+d, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i] * G[(i+d)%N, i]
        """

        # Compute the four-point green's function using the ground truth
        # G_delta_0_G_delta_0 = mean_i G_{i+d, i} G_{i+d, i}
        G = M_inv # [N, N]

        # Block concat: [[G, -G], [-G, G]] for G of shape [N, N]
        G = torch.cat([
            torch.cat([G, -G], dim=1),
            torch.cat([-G, G], dim=1)
        ], dim=0)  # [2N, 2N]

        N = G.shape[0]
        Ltau, Ly, Lx = 2 * self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, idx] * G[i, i]

        GG = result.mean(dim=0) # [Ltau * Ly * Lx]
        return GG.view(Ltau, Ly, Lx)[:self.Ltau]


    def G_delta_0_G_0_delta_groundtruth_ext(self, M_inv):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[i+d, i] * G[i+d, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i] * G[(i+d)%N, i]
        """

        # Compute the four-point green's function using the ground truth
        # G_delta_0_G_delta_0 = mean_i G_{i+d, i} G_{i+d, i}
        G = M_inv # [N, N]

        # Block concat: [[G, -G], [-G, G]] for G of shape [N, N]
        G = torch.cat([
            torch.cat([G, -G], dim=1),
            torch.cat([-G, G], dim=1)
        ], dim=0)  # [2N, 2N]

        N = G.shape[0]
        Ltau, Ly, Lx = 2 * self.Ltau, self.Ly, self.Lx
        result = torch.empty((N, N), dtype=G.dtype, device=G.device)
        for i in range(N):
            for d in range(N):
                tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                idx = ravel_multi_index(
                    ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                    (Ltau, Ly, Lx)
                )

                result[i, d] = G[idx, i] * G[i, idx]

        GG = result.mean(dim=0) # [Ltau * Ly * Lx]
        return GG.view(Ltau, Ly, Lx)[:self.Ltau]


    # -------- Fermionic obs methods --------
    def spsm_k(self):
        # Observables
        # !zspsm(imj) = zspsm(imj) + grupc(i,j)*grup(i,j)
        # ! up-down not the same anymore
        # zspsm(imj) = zspsm(imj) + chalf * ( grupc(i,j)*grdn(i,j) + grdnc(i,j)*grup(i,j) )
        # zszsz(imj) = zszsz(imj) + chalf*chalf*( (grupc(i,i)-grdnc(i,i))*(grupc(j,j)-grdnc(j,j)) + ( grupc(i,j)*grup(i,j)+grdnc(i,j)*grdn(i,j) ) )

        # Defs
        # !grup (i,j) = < c_i c^+_j >
        # !grupc (i,j) = < c^+_i c_j >

        # ! get grupc
        # do i = 1, ndim
        #     do j = 1, ndim
        #         grupc(j,i) = - grup(i,j)
        #     end do
        #     grupc(i,i) = grupc(i,i) + cone
        # end do

        # do i = 1,ndim
        # nx_i = list(i,1)
        # ny_i = list(i,2)
        # xi = 1.d0
        # if ( mod(nx_i,2) .ne. mod(ny_i,2) ) xi = -1.d0
        # do j = 1,ndim
        #     nx_j = list(j,1)
        #     ny_j = list(j,2)
        #     xj = 1.d0
        #     if ( mod(nx_j,2) .ne. mod(ny_j,2) ) xj = -1.d0
        #     grdn (i,j) = dcmplx(xi*xj, 0.d0)*dconjg ( grupc(i,j) )
        #     grdnc(i,j) = dcmplx(xi*xj, 0.d0)*dconjg ( grup (i,j) )
        # enddo
        # enddo

        """
        zsp_i sm_j = <c^+_i c_j><c_i c^+_j> = (delta_ij  - <c_j c^+_i>)<c_i c^+_j>
        = grupc(i, j) * grup(i, j) = (- grup(j, i) + delta_ij) * grup(i, j)
        Delta = i - j
        = - grup(0, D) * grup (D, 0) + grup(0, 0) * delta_{D, 0}
        = - GG_D00D + G_D0 * delta_{D, 0}

        spsm: [Lkx, Lky]
        """
        spsm = -self.G_delta_0_G_0_delta_ext()[0]  # [Ly, Lx]
        spsm[0, 0] += self.G_delta_0_ext()[0, 0, 0]
        # spsm = spsm.real
        spsm_k = torch.fft.ifft2(spsm, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        spsm_k = self.reorder_fft_grid2(spsm_k).permute(1, 0)  # [Lx, Ly]
        return spsm_k

    def szsz(self):
        # zszsz(imj) = zszsz(imj) + chalf*chalf*( (grupc(i,i)-grdnc(i,i))*(grupc(j,j)-grdnc(j,j)) + ( grupc(i,j)*grup(i,j)+grdnc(i,j)*grdn(i,j) ) )
        # -> 1/2 * (grupc(i,j) * grup(i,j)) = 1/2 * (- grup(j, i) + delta_ij) * grup(i, j)
        szsz = -self.G_delta_0_G_0_delta_ext()
        szsz[0, 0, 0] = szsz[0, 0, 0] + self.G_delta_0_ext()[0, 0, 0]
        return 0.5 * szsz

    def get_fermion_obsr(self, bosons, eta):
        """
        bosons: [bs, 2, Ltau, Ly, Lx] tensor of boson fields

        Returns:
            spsm: [bs, Ltau=1, Ly, Lx] tensor, spsm[i, j, tau] = <c^+_i c_j> * <c_i c^+_j>
            szsz: [bs, Ltau=1, Ly, Lx] tensor, szsz[i, j, tau] = <c^+_i c_i> * <c^+_j c_j>
        """
        bs, _, Lx, Ly, Ltau = bosons.shape
        spsm = torch.zeros((bs, Lx, Ly), dtype=self.cdtype, device=self.device)
        szsz = torch.zeros((bs, Lx, Ly), dtype=self.cdtype, device=self.device)
        
        ky = torch.fft.fftfreq(self.Ly)
        kx = torch.fft.fftfreq(self.Lx)
        ks = torch.stack(torch.meshgrid(ky, kx, indexing='ij'), dim=-1) # [Ly, Lx, 2]
        
        ks = ks.permute(2, 0, 1)  # [2, Ly, Lx]
        ks_ordered = self.reorder_fft_grid2(ks).permute(2, 1, 0)  # [Lx, Ly, 2]
        ks_ordered = torch.flip(ks_ordered, dims=[-1])  # [Lx, Ly, 2]
        for b in range(bs):
            boson = bosons[b].unsqueeze(0)  # [1, 2, Ltau, Ly, Lx]

            self.set_eta_G_eta(boson, eta)
            spsm[b] = self.spsm_k()
            szsz[b] = 0.5 * spsm[b]

        obsr = {}
        obsr['spsm'] = spsm
        obsr['szsz'] = szsz
        obsr['ks'] = ks_ordered
        return obsr


def test_green_functions():
    hmc = HmcSampler()
    hmc.Lx = 2
    hmc.Ly = 2
    hmc.Ltau = 4

    hmc.bs = 3
    hmc.reset()
    hmc.initialize_boson_pi_flux_randn_matfree()
    boson = hmc.boson[1].unsqueeze(0)
    hmc.bs = 1

    se = StochaticEstimator(hmc)
    se.Nrv = 100_000  # bs > 10000 will fail on _C.mh_vec, due to grid = {Ltau, bs}.
    se.Nrv = 200  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent
    
    se.test_orthogonality(se.random_vec_bin())
    # se.test_orthogonality(se.random_vec_norm())
    se.test_fft_negate_k3()

    # Compute Green prepare
    eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]
    # eta = se.random_vec_norm().to(torch.complex64)  # [Nrv, Ltau * Ly * Lx]

    se.set_eta_G_eta_debug(boson, eta)
    Gij_gt = se.G_groundtruth(boson)

    # Test Green
    G_stoch = se.G_delta_0()
    G_stoch_primitive = se.G_delta_0_primitive()
    torch.testing.assert_close(G_stoch.real, G_stoch_primitive.real, rtol=1e-2, atol=5e-2)
    
    G_gt = se.G_delta_0_groundtruth(Gij_gt)
    torch.testing.assert_close(G_gt.real, G_stoch_primitive.real, rtol=1e-2, atol=5e-2)

    # Test Green extended
    G_stoch_ext = se.G_delta_0_ext()
    G_stoch_primitive_ext = se.G_delta_0_primitive_ext()
    torch.testing.assert_close(G_stoch_ext.real, G_stoch_primitive_ext.real, rtol=1e-2, atol=5e-2)

    G_gt_ext = se.G_delta_0_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(G_gt_ext.real, G_stoch_primitive_ext.real, rtol=1e-2, atol=5e-2)

    # Test Green four-point
    GG_stoch_primitive = se.G_delta_0_G_delta_0_primitive()
    GG_gt = se.G_delta_0_G_delta_0_groundtruth(Gij_gt)
    GG_stoch = se.G_delta_0_G_delta_0()
    torch.testing.assert_close(GG_stoch_primitive, GG_gt, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(GG_stoch, GG_stoch_primitive, rtol=1e-2, atol=2e-2)

    # Test Green four-point GG_D0D0 extended
    GG_ext = se.G_delta_0_G_delta_0_ext()
    GG_primitive_ext = se.G_delta_0_G_delta_0_primitive_ext()
    GG_gt_ext = se.G_delta_0_G_delta_0_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(GG_primitive_ext, GG_gt_ext, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(GG_primitive_ext, GG_ext, rtol=1e-2, atol=2e-2)

    # Test Green four-point GG_DD00 extended
    GG_ext = se.G_delta_delta_G_0_0_ext()
    GG_gt_ext = se.G_delta_delta_G_0_0_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(GG_gt_ext, GG_ext, rtol=1e-2, atol=2e-2)

    # Test Green four-point GG_DD00 extended
    GG_ext = se.G_delta_0_G_0_delta_ext()
    GG_gt_ext = se.G_delta_0_G_0_delta_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(GG_gt_ext, GG_ext, rtol=1e-2, atol=2e-2)
 

    print("✅ All assertions pass!")


def test_fermion_obsr():
    hmc = HmcSampler()
    hmc.Lx = 6
    hmc.Ly = 6
    hmc.Ltau = 10

    hmc.bs = 5
    hmc.reset()
    hmc.initialize_boson_pi_flux_randn_matfree()

    se = StochaticEstimator(hmc)
    se.Nrv = 200  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent
    se.init_cuda_graph()

    # Compute Green prepare
    eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]

    bosons = hmc.boson
    if se.cuda_graph:
        obsr = se.graph_runner(bosons, eta)
    else:
        obsr = se.get_fermion_obsr(bosons, eta)

    obsr_ref = se.get_fermion_obsr(bosons, eta)
    torch.testing.assert_close(obsr['spsm'], obsr_ref['spsm'], rtol=1e-2, atol=5e-2)
    print()

    # ---------- Benchmark vs dqmc ---------- #
    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    J = 0.5
    bs = 5
    assert bs == hmc.bs, "Batch size mismatch."
    input_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/benchmark_6x6x10_bs5/hmc_check_point_6x10/"
    input_folder = "/users/4/fengx463/hmc/qed_fermion/qed_fermion/post_processors/fermi_bench/"
    hmc_filename = f"/stream_ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_6000_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.05_N_leapfrog_4_m_1_step_6000.pt"

    # Parse to get specifics
    path_parts = hmc_filename.split('/')
    filename = path_parts[-1]
    filename_parts = filename.split('_')
    specifics = '_'.join(filename_parts[1:]).replace('.pt', '')
    print(f"Parsed specifics: {specifics}")

    parts = hmc_filename.split('_')
    jtau_index = parts.index('Jtau')  # Find position of 'Jtau'
    jtau_value = float(parts[jtau_index + 1])   # Get the next element
    
    # Load and write
    # [seq, Ltau * Ly * Lx * 2]
    boson_seq = torch.load(input_folder + hmc_filename)
    # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
    print(f'Loaded: {input_folder + hmc_filename}')        
    
    # Extract Nstep and Nstep_local from filenames
    # hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    # end = int(hmc_match.group(1))
    end = 6000

    start = 5999
    sample_step = 1
    seq_idx = set(list(range(start, end, sample_step)))

    # Write result to file

    filtered_seq = [(i, boson) for i, boson in enumerate(boson_seq) if i in seq_idx]
    spsm = torch.zeros((len(filtered_seq), bs, Lx, Ly), dtype=hmc.cdtype)
    for i, boson in filtered_seq:
        print(f"boson shape: {boson[1].shape}, dtype: {boson[1].dtype}, device: {boson[1].device}")

        if se.cuda_graph:
            obsr = se.graph_runner(bosons, eta)
        else:
            obsr = se.get_fermion_obsr(bosons, eta)
        spsm[i-start] = obsr['spsm']  # [bs, Lx, Ly]
    
    ks = obsr['ks']  # [Lx, Ly, 2]

    # Linearize
    spsm_mean = spsm.mean(dim=(0))  # [bs, Lx, Ly]
    spsm_mean = spsm_mean.permute((0, 2, 1)).reshape(bs, -1)  # Ly*Lx
    ks = ks.permute((1, 0, 2)).reshape(-1, 2)  # Ly*Lx, but displayed as (kx, ky)
    print("ks (flattened):", ks)

    for b in range(bs):
        print(f"spsm_mean (flattened) bid:{b}: ", spsm_mean[b].real)

    output_dir = os.path.join(script_path, "post_processors/fermi_bench")
    os.makedirs(output_dir, exist_ok=True)
    for b in range(bs):
        data = torch.stack([ks[:, 0], ks[:, 1], spsm_mean[b].real], dim=1).cpu().numpy()
        output_file = os.path.join(output_dir, f"spsm_k_b{b}.txt")
        # Save as text, columns: kx, ky, spsm
        np.savetxt(output_file, data, fmt="%.8f", comments='')
        print(f"Saved: {output_file}")

    print("✅ All assertions pass!")


if __name__ == "__main__":  
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # test_green_functions()
    
    test_fermion_obsr()


    # hmc = HmcSampler()
    # hmc.Lx = 2
    # hmc.Ly = 2
    # hmc.Ltau = 4

    # hmc.bs = 3
    # hmc.reset()
    # hmc.initialize_boson_pi_flux_randn_matfree()
    # boson = hmc.boson[1].unsqueeze(0)
    # hmc.bs = 1

    # se = StochaticEstimator(hmc)
    # se.Nrv = 200  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent

    # # Compute Green prepare
    # eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]
    # se.set_eta_G_eta(boson, eta)

