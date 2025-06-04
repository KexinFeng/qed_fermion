import numpy as np
import torch
import os 
import sys

from tqdm import tqdm
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.fermion_obsr_graph_runner import FermionObsrGraphRunner
from qed_fermion.utils.util import ravel_multi_index, unravel_index, device_mem

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

if torch.cuda.is_available():
    from qed_fermion import _C 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = (4, 8)
print(f"BLOCK_SIZE: {BLOCK_SIZE}")

Nrv = float(os.getenv("Nrv", '10'))
print(f"Nrv: {Nrv}")
max_iter_se = float(os.getenv("max_iter_se", '200'))
print(f"max_iter_se: {max_iter_se}")

class StochaticEstimator:
    # This function computes the fermionic green's function, mainly four-point function. The aim is to surrogate the postprocessing of the HMC boson samples. For a given boson, the M(boson) is the determined, so is M_inv aka green's function.

    # So the algorithm is to generate Nrv iid random variables and utilize the orthogonaliaty of the random variables expectation, to estimate M_inv
    #   G_ij ~ (G eta)_i eta_j
    #   G_ij G_kl ~ (G eta)_i eta_j (G eta')_k eta'_l

    # So one method to generate Nrv eta vector: random_vec 
    # One method to estimate the four-point green's function: four_point_green
    # One method to compute the obs like spsm szsz, etc.

    def __init__(self, hmc, cuda_graph_se=False):
        self.hmc_sampler = hmc
        self.Nrv = Nrv
        self.max_iter_se = max_iter_se

        self.Lx = hmc.Lx
        self.Ly = hmc.Ly    
        self.Ltau = hmc.Ltau
        self.dtau = hmc.dtau

        self.Vs = hmc.Vs

        self.cuda_graph_se = cuda_graph_se
        self.device = hmc.device
        self.dtype = hmc.dtype
        self.cdtype = hmc.cdtype
        
        self.graph_runner = FermionObsrGraphRunner(self)
        self.graph_memory_pool = hmc.graph_memory_pool

        # init
        if hmc.precon_csr is None:
            hmc.reset_precon()

    def init_cuda_graph(self):
        hmc = self.hmc_sampler
        # Capture
        if self.cuda_graph_se:
            print("Initializing CUDA graph for get_fermion_obsr.........")
            print(f"Lx_{self.Lx}, Ltau_{self.Ltau}, Nrv_{self.Nrv}, max_iter_se_{self.max_iter_se}")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init se_graph: {d_mem_str}")
            dummy_eta = torch.zeros((self.Nrv, self.Ltau * self.Vs), device=hmc.device, dtype=hmc.cdtype)
            dummy_bosons = torch.zeros((hmc.bs, 2, self.Lx, self.Ly, self.Ltau), device=hmc.device, dtype=hmc.dtype)
            self.graph_memory_pool = self.graph_runner.capture(
                                        dummy_bosons, 
                                        dummy_eta, 
                                        max_iter_se=self.max_iter_se,
                                        graph_memory_pool=self.graph_memory_pool)
            print(f"get_fermion_obsr CUDA graph initialization complete")
            d_mem_str, d_mem3 = device_mem()
            print(f"After init se_graph: {d_mem_str}, incr.by: {d_mem3 - d_mem2:.2f} MB\n") 
            print('')

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
    def reorder_fft_grid2(tensor2d, dims=(-2, -1)):
        """Reorder the last two axes of a tensor from FFT-style to ascending momentum order."""
        Ny, Nx = tensor2d.shape[dims[0]], tensor2d.shape[dims[1]]
        return torch.roll(tensor2d, shifts=(Ny // 2, Nx // 2), dims=dims)


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
        print("max_pcg_iter:", cnt[:5] if not self.hmc_sampler.cuda_graph else "cuda_graph_on")
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
    
    def G_groundtruth_sparse(self, boson):
        res = self.hmc_sampler.get_M_batch(boson)
        M = res[-1].to_sparse_csr()
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


    def G_delta_0_groundtruth_ext_fft(self, M_inv, debug=False):
        """
        Given G of shape [N, N] (N = Lx*Ly*Ltau), compute tensor of shape [N, N] where
        result[i, d] = G[idx, i], with periodic boundary conditions.

        Returns:
            result: [N, N] tensor, result[i, d] = G[(i+d)%N, i]
        """
        G = M_inv  # [N, N]: N = Ltau * Ly * Lx
        
        if debug:
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
            G_mean_ref = result.mean(dim=0)  # [Ltau * Ly * Lx]
            G_mean_ref = G_mean_ref.view(Ltau2, Ly, Lx)[:self.Ltau]


        # FFT-based implementation for spatial correlations at tau=0
        # Reshape G to [2*Ltau, Ly, Lx, 2*Ltau, Ly, Lx] conceptually
        G_reshaped = M_inv.view(self.Ltau, Ly, Lx, self.Ltau, Ly, Lx)

        # fft
        # mean_r G[:, r, : r + d] = G(d)
        # G[:, r, :, r'] = G[:, k, :, k'] e^{ikr + ik'r'}
        # sum_r G[:, r, : r + d] = sum_r e^{ikr+ ik'r + ik'd} G[:, k, :, k']
        # = G[:, -k, :, k] e^{ikd}
        # G(d) -> G[:, r, :, r'] 

        # FFT-based implementation for spatial correlations G(d)
        G_fft = torch.fft.fftn(G_reshaped, dim=(1, 2, 4, 5), norm="forward")  # [2*Ltau, Ly, Lx, 2*Ltau, Ly, Lx]

        # Now, for each tau, tau', we want G[:, -k, :, k] (i.e., k = k')
        # So, for each kx, ky, select G[:, -ky, -kx, :, ky, kx]
        Ly, Lx = self.Ly, self.Lx

        # Prepare frequency indices
        # ky = torch.fft.fftfreq(Ly, device=G_fft.device)
        # kx = torch.fft.fftfreq(Lx, device=G_fft.device)
        ky_idx = torch.arange(Ly, device=G_fft.device)
        kx_idx = torch.arange(Lx, device=G_fft.device)

        # # For each (ky, kx), get the index for -ky, -kx (modulo Ly, Lx)
        # ky_neg_idx = (-ky_idx) % Ly
        # kx_neg_idx = (-kx_idx) % Lx

        # Use advanced indexing to select G[:, -ky, -kx, :, ky, kx]
        # We'll build a meshgrid for all (ky, kx)
        ky_grid, kx_grid = torch.meshgrid(ky_idx, kx_idx, indexing='ij')  # [Ly, Lx]
        ky_neg_grid = (-ky_grid) % Ly
        kx_neg_grid = (-kx_grid) % Lx

        if debug:
            # Now, for each tau, tau', select G_fft[:, ky_neg, kx_neg, :, ky, kx]
            G_fft_diag = torch.empty((self.Ltau, Ly, Lx), dtype=G_fft.dtype, device=G_fft.device)
            for tau in range(self.Ltau):
                # G_fft[tau, ky_neg, kx_neg, tau, ky, kx]
                G_fft_diag[tau] = G_fft[tau, ky_neg_grid, kx_neg_grid, tau, ky_grid, kx_grid]

        # tau_idx = torch.arange(self.Ltau, device=G_fft.device)
        # Instead of just ky_grid, kx_grid, get meshgrid for (tau_idx, ky_idx, kx_idx)
        tau_idx = torch.arange(self.Ltau, device=G_fft.device)
        ky_idx = torch.arange(Ly, device=G_fft.device)
        kx_idx = torch.arange(Lx, device=G_fft.device)
        tau_grid, ky_grid, kx_grid = torch.meshgrid(tau_idx, ky_idx, kx_idx, indexing='ij')
        # Now tau_grid, ky_grid, kx_grid are all [Ltau, Ly, Lx] and enumerate all combinations
        ky_neg_grid = (-ky_grid) % Ly
        kx_neg_grid = (-kx_grid) % Lx
        G_fft_diag_adv_idx = G_fft[tau_grid, ky_neg_grid, kx_neg_grid, tau_grid, ky_grid, kx_grid]  # [Ltau, Ly, Lx]
        
        if debug:
            torch.testing.assert_close(G_fft_diag, G_fft_diag_adv_idx, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)

        # Now, IFFT to get G(d)
        G_d = torch.fft.ifft2(G_fft_diag, s=(Ly, Lx), norm="backward")  # [2*Ltau, Ly, Lx]
        G_mean_fft = G_d.mean(dim=0) * Lx*Ly  # [Ly, Lx]

        if debug:
            # Verify equivalence with reference implementation
            torch.testing.assert_close(G_mean_ref[0].real, G_mean_fft.real, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)
            torch.testing.assert_close(G_mean_ref[0], G_mean_fft, rtol=1e-3, atol=1e-3, equal_nan=True, check_dtype=False)

        return G_mean_fft  # Return tau=0 slice: [Ly, Lx]


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

    def G_delta_0_G_0_delta_groundtruth_ext_fft(self, M_inv, debug=False):
        """
        FFT-based implementation of the four-point Green's function:
        result[i, d] = G[(i+d)%N, i] * G[i, (i+d)%N], with periodic boundary conditions.

        Returns:
            [Ly, Lx] tensor (tau=0 slice)
        """
        G = M_inv  # [N, N]: N = Ltau * Ly * Lx

        if debug:
            Ltau2 = 2 * self.Ltau
            Lx = self.Lx
            Ly = self.Ly

            # Block concat: [[G, -G], [-G, G]] for G of shape [N, N]
            G_block = torch.cat([
                torch.cat([G, -G], dim=1),
                torch.cat([-G, G], dim=1)
            ], dim=0)  # [2N, 2N]

            N = G_block.shape[0]
            Ltau, Ly, Lx = 2*self.Ltau, self.Ly, self.Lx
            result = torch.empty((N, N), dtype=G_block.dtype, device=G_block.device)
            for i in range(N):
                for d in range(N):
                    tau, y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))
                    dtau, dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ltau, Ly, Lx))

                    idx = ravel_multi_index(
                        ((tau + dtau) % Ltau, (y + dy) % Ly, (x + dx) % Lx),
                        (Ltau, Ly, Lx)
                    )

                    result[i, d] = G_block[idx, i] * G_block[i, idx]
            GG_ref = result.mean(dim=0)  # [Ltau * Ly * Lx]
            GG_ref = GG_ref.view(Ltau2, Ly, Lx)[:self.Ltau]


        # FFT-based implementation
        # mean_r G[:, r+d, :, r] * G[:, r, : r+d] = G(d)
        # G[:, r, :, r'] = G[:, k, :, k'] e^{ikr + ik'r'}
        # sum_r G[:, k'', :, k'''] * G[:, k, :, k'] e^{ikr + ik'r + ik'd + ik''d + ik''r + ik'''r} 
        # = G[:, k'', :, k'''] * G[:, k, :, k'] delta(k+k'+k''+k''') e^{i(k'+k'')d}
        # = G[:, k'', :, -k-k'-k''] * G[:, k, :, k'] e^{i(k'+k'')d}
        #sum_{k, k'} G[:, -k'+p, :, -k-p] * G[:, k, :, k'] e^{i * p * d}

        G_reshaped = M_inv.view(self.Ltau, Ly, Lx, self.Ltau, Ly, Lx)
        G_fft = torch.fft.fftn(G_reshaped, dim=(1, 2, 4, 5), norm="forward")  # [Ltau, Ly, Lx, Ltau, Ly, Lx]

        Ly, Lx = self.Ly, self.Lx
        result_fft = torch.zeros((self.Ltau, Ly, Lx), dtype=G_fft.dtype, device=G_fft.device)

        # # ---- vectorized version ---- #
        # # Prepare all indices including tau
        # tau_idx = torch.arange(self.Ltau, device=G_fft.device)
        # ky_idx = torch.arange(Ly, device=G_fft.device)
        # kx_idx = torch.arange(Lx, device=G_fft.device)
        # py_idx = torch.arange(Ly, device=G_fft.device)
        # px_idx = torch.arange(Lx, device=G_fft.device)
        
        # # Create meshgrid for all dimensions at once
        # tau_grid, ky_grid1, kx_grid1, ky_grid2, kx_grid2, py_grid, px_grid = torch.meshgrid(
        #     tau_idx, ky_idx, kx_idx, ky_idx, kx_idx, py_idx, px_idx, indexing='ij'
        # )  # [Ltau, Ly, Lx, Ly, Lx, Ly, Lx]

        # # Vectorized computation for all tau and p = (py, px)
        # # sum_{k, k'} G[:, -k'+p, :, -k-p] * G[:, k, :, k'] e^{i * p * d}
        # G1 = G_fft[tau_grid, 
        #        (-ky_grid2 + py_grid) % Ly, 
        #        (-kx_grid2 + px_grid) % Lx, 
        #        tau_grid,
        #        (-ky_grid1 - py_grid) % Ly, 
        #        (-kx_grid1 - px_grid) % Lx]
        # G2 = G_fft[tau_grid, ky_grid1, kx_grid1,
        #            tau_grid, ky_grid2, kx_grid2]

        # # sum over k, k' dimensions (dimensions 1,2,3,4)
        # result_fft = (G1 * G2).sum(dim=(1, 2, 3, 4))  # [Ltau, Ly, Lx]


        # ---- semi vectorized version ---- #
        # Vectorized computation over all (py, px) at once
        # G_fft_tau: [Ly, Lx, Ly, Lx] for fixed tau
        for tau in tqdm(range(self.Ltau)):
            # G_fft_tau: [Ly, Lx, Ly, Lx] for fixed tau
            G_fft_tau = G_fft[tau, :, :, tau, :, :]  # [Ly, Lx, Ly, Lx]

            # Prepare k, k' indices
            ky_idx = torch.arange(Ly, device=G_fft.device)
            kx_idx = torch.arange(Lx, device=G_fft.device)
            py_idx = torch.arange(Ly, device=G_fft.device)
            px_idx = torch.arange(Lx, device=G_fft.device)
            
            ky_grid1, kx_grid1, ky_grid2, kx_grid2, py_grid, px_grid = torch.meshgrid(
                ky_idx, kx_idx, ky_idx, kx_idx, py_idx, px_idx, indexing='ij'
            )  # [Ly, Lx, Ly, Lx, Ly, Lx]

            # Vectorized computation for all p = (py, px)
            #sum_{k, k'} G[:, -k'+p, :, -k-p] * G[:, k, :, k'] e^{i * p * d}
            G1 = G_fft_tau[(-ky_grid2 + py_grid) % Ly, 
                           (-kx_grid2 + px_grid) % Lx, 
                           (-ky_grid1 - py_grid) % Ly, 
                           (-kx_grid1 - px_grid) % Lx]
            G2 = G_fft_tau[ky_grid1, kx_grid1, ky_grid2, kx_grid2]

            # sum over k, k' dimensions (first 4 dimensions)
            val = (G1 * G2).sum(dim=(0, 1, 2, 3))  # [Ly, Lx]
            result_fft[tau] = val


        # ----- Loop version ---- #
        # for tau in range(self.Ltau):
        #     # G_fft_tau: [Ly, Lx, Ly, Lx] for fixed tau
        #     G_fft_tau = G_fft[tau, :, :, tau, :, :]  # [Ly, Lx, Ly, Lx]

        #     # Prepare k, k' indices
        #     ky_idx = torch.arange(Ly, device=G_fft.device)
        #     kx_idx = torch.arange(Lx, device=G_fft.device)
        #     ky_grid1, kx_grid1, ky_grid2, kx_grid2 = torch.meshgrid(
        #         ky_idx, kx_idx, ky_idx, kx_idx, indexing='ij'
        #     )  # [Ly, Lx, Ly, Lx]

        #     # For each p = (py, px)
        #     for py in range(Ly):
        #         for px in range(Lx):
        #             #sum_{k, k'} G[:, -k'+p, :, -k-p] * G[:, k, :, k'] e^{i * p * d}
        #             G1 = G_fft_tau[(-ky_grid2 + py) % Ly, 
        #                            (-kx_grid2 + px) % Lx, 
        #                            (-ky_grid1 - py) % Ly, 
        #                            (-kx_grid1 - px) % Lx]
        #             G2 = G_fft_tau[ky_grid1, kx_grid1, ky_grid2, kx_grid2]

        #             # sum over k, k'
        #             val = (G1 * G2).sum()
        #             result_fft[tau, py, px] = val

        # Now, for each tau, do iFFT over (py, px) to get G(d)
        G_d_fft = torch.fft.ifft2(result_fft, s=(Ly, Lx), norm="backward")  # [Ltau, Ly, Lx]
        G_mean_fft = G_d_fft.mean(dim=0) * Lx * Ly  # [Ly, Lx]

        if debug:
            torch.testing.assert_close(GG_ref[0].real, G_mean_fft.real, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)
            torch.testing.assert_close(GG_ref[0], G_mean_fft, rtol=1e-3, atol=1e-3, equal_nan=True, check_dtype=False)

        return G_mean_fft  # tau=0 slice: [Ly, Lx]


    # -------- Fermionic obs methods --------
    def spsm_r(self, GD0_G0D, GD0):
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
        = - grup (D, 0) * grup(0, D) + grup(0, 0) * delta_{D, 0}
        = - GG_D00D + G_D0 * delta_{D, 0}

        spsm: [Ly, Lx]
        """
        spsm = -GD0_G0D[0]  # [Ly, Lx]
        spsm[0, 0] += GD0[0, 0, 0]
        spsm = spsm.real
        return spsm
    
    def spsm_r_minus_bg(self, GD0_G0D, GD0):
        """
        spsm: [Ly, Lx]
        """
        spsm = self.spsm_r(GD0_G0D, GD0)
        spsm -= (GD0[0, 0, 0]**2).abs()
        return spsm
    
    def spsm_k(self, spsm_r):
        """
        spsm_r: [Ly, Lx]
        spsm_k: [Lky, Lkx]
        """
        spsm_k = torch.fft.ifft2(spsm_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        spsm_k = self.reorder_fft_grid2(spsm_k)  # [Ly, Lx]
        return spsm_k
    
    def get_ks_ordered_xy(self):
        """
        Returns:
            ks_ordered: [Lx, Ly, 2] tensor, where ks_ordered[:, :, 0] = kx and ks_ordered[:, :, 1] = ky
            kx = 2 * pi * n / Lx, ky = 2 * pi * m / Ly
            n = -Lx/2, ..., -1, 0, 1, ..., Lx/2-1
            m = -Ly/2, ..., -1, 0, 1, ..., Ly/2-1
        """
        ky = torch.fft.fftfreq(self.Ly)
        kx = torch.fft.fftfreq(self.Lx)
        ks = torch.stack(torch.meshgrid(ky, kx, indexing='ij'), dim=-1) # [Ly, Lx, 2]
        
        ks = ks.permute(2, 0, 1)  # [2, Ly, Lx]
        ks_ordered = self.reorder_fft_grid2(ks).permute(2, 1, 0)  # [Lx, Ly, 2]
        ks_ordered = torch.flip(ks_ordered, dims=[-1])  # [Lx, Ly, 2]
        return ks_ordered
    
    def get_ks_ordered(self):
        ky = torch.fft.fftfreq(self.Ly)
        kx = torch.fft.fftfreq(self.Lx)
        ks = torch.stack(torch.meshgrid(ky, kx, indexing='ij'), dim=-1) # [Ly, Lx, (ky, kx)]

        ks_ordered = self.reorder_fft_grid2(ks, dims=(0, 1))  # [Ly, Lx, 2]
        return ks_ordered     

    def szsz(self):
        # zszsz(imj) = zszsz(imj) + chalf*chalf*( (grupc(i,i)-grdnc(i,i))*(grupc(j,j)-grdnc(j,j)) + ( grupc(i,j)*grup(i,j)+grdnc(i,j)*grdn(i,j) ) )
        # -> 1/2 * (grupc(i,j) * grup(i,j)) = 1/2 * (- grup(j, i) + delta_ij) * grup(i, j)
        szsz = -self.G_delta_0_G_0_delta_ext()
        szsz[0, 0, 0] = szsz[0, 0, 0] + self.G_delta_0_ext()[0, 0, 0]
        return 0.5 * szsz

    def get_fermion_obsr(self, bosons, eta):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields

        Returns:
            spsm: [bs, Ly, Lx] tensor, spsm[i, j, tau] = <c^+_i c_j> * <c_i c^+_j>
            szsz: [bs, Ly, Lx] tensor, szsz[i, j, tau] = <c^+_i c_i> * <c^+_j c_j>
        """
        bs, _, Lx, Ly, Ltau = bosons.shape
        spsm_r = torch.zeros((bs, Ly, Lx), dtype=self.dtype, device=self.device)
        spsm_k_abs = torch.zeros((bs, Ly, Lx), dtype=self.dtype, device=self.device)
        # szsz = torch.zeros((bs, Lx, Ly), dtype=self.dtype, device=self.device)
        
        for b in range(bs):
            boson = bosons[b].unsqueeze(0)  # [1, 2, Ltau, Ly, Lx]

            self.set_eta_G_eta(boson, eta)
            GD0_G0D = self.G_delta_0_G_0_delta_ext() # [Ly, Lx]
            GD0 = self.G_delta_0_ext() # [Ly, Lx]

            spsm_r_per_b = self.spsm_r(GD0_G0D, GD0)  # [Ly, Lx]
            # spsm_r[b] = spsm_r_per_b
            spsm_r[b] = self.spsm_r_minus_bg(GD0_G0D, GD0)  # [Ly, Lx]
            spsm_k_abs[b] = self.spsm_k(spsm_r_per_b).abs()  # [Ly, Lx]

            # szsz[b] = 0.5 * spsm[b]

        obsr = {}
        obsr['spsm_r'] = spsm_r
        obsr['spsm_k_abs'] = spsm_k_abs
        return obsr
    
    def get_fermion_obsr_groundtruth(self, bosons):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields

        Returns:
            spsm: [bs, Ly, Lx] tensor, spsm[i, j, tau] = <c^+_i c_j> * <c_i c^+_j>
            szsz: [bs, Ly, Lx] tensor, szsz[i, j, tau] = <c^+_i c_i> * <c^+_j c_j>
        """
        bs, _, Lx, Ly, Ltau = bosons.shape
        spsm_r = torch.zeros((bs, Ly, Lx), dtype=self.dtype, device=self.device)
        spsm_k_abs = torch.zeros((bs, Ly, Lx), dtype=self.dtype, device=self.device)
        # szsz = torch.zeros((bs, Lx, Ly), dtype=self.dtype, device=self.device)
        
        for b in range(bs):
            boson = bosons[b].unsqueeze(0)  # [1, 2, Ltau, Ly, Lx]

            # self.set_eta_G_eta(boson, eta)
            # GD0_G0D = self.G_delta_0_G_0_delta_ext() # [Ly, Lx]
            # GD0 = self.G_delta_0_ext() # [Ly, Lx]
            Gij_gt = self.G_groundtruth(boson)
            GD0 = self.G_delta_0_groundtruth_ext_fft(Gij_gt)
            GD0_G0D = self.G_delta_0_G_0_delta_groundtruth_ext_fft(Gij_gt)

            spsm_r_per_b = self.spsm_r(GD0_G0D, GD0)  # [Ly, Lx]
            # spsm_r[b] = spsm_r_per_b
            spsm_r[b] = self.spsm_r_minus_bg(GD0_G0D, GD0)  # [Ly, Lx]
            spsm_k_abs[b] = self.spsm_k(spsm_r_per_b).abs()  # [Ly, Lx]

            # szsz[b] = 0.5 * spsm[b]

        obsr = {}
        obsr['spsm_r'] = spsm_r
        obsr['spsm_k_abs'] = spsm_k_abs
        return obsr



if __name__ == "__main__":  
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # test_green_functions()
    
    # test_fermion_obsr()


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

