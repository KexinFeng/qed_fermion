import math
import torch
import os 
import sys

from tqdm import tqdm
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.fermion_obsr_graph_runner import FermionObsrGraphRunner, GEtaGraphRunner, L0GraphRunner, SpsmGraphRunner, T1GraphRunner, T21GraphRunner, T2GraphRunner, T4GraphRunner, TvvGraphRunner
from qed_fermion.utils.util import ravel_multi_index, unravel_index, device_mem, tensor_memory_MB
import torch.nn.functional as F

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

if torch.cuda.is_available():
    from qed_fermion import _C 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = (4, 8)
print(f"BLOCK_SIZE: {BLOCK_SIZE}")

Nrv = int(os.getenv("Nrv", '20'))
print(f"Nrv: {Nrv}")
max_iter_se = int(os.getenv("max_iter_se", '100'))
# print(f"max_iter_se: {max_iter_se}")
precon_on = int(os.getenv("precon", '1')) == 1
print(f"precon_on: {precon_on}")

capture_fermion_obsr = False

# Initialize a simple object to hold parameters
class Params: pass

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
        self.Nrv = int(Nrv)
        self.max_iter_se = max_iter_se
        self.num_inner_loops = 200

        self.Lx = hmc.Lx
        self.Ly = hmc.Ly    
        self.Ltau = hmc.Ltau
        self.dtau = hmc.dtau

        self.Vs = hmc.Vs

        self.cuda_graph_se = cuda_graph_se
        self.device = hmc.device
        self.dtype = hmc.dtype
        self.cdtype = hmc.cdtype
        
        self.graph_memory_pool = hmc.graph_memory_pool
        self.graph_runner = FermionObsrGraphRunner(self)
        self.G_eta_graph_runner = GEtaGraphRunner(self)
        self.L0_graph_runner = L0GraphRunner(self)
        self.spsm_graph_runner = SpsmGraphRunner(self)
        self.T1_graph_runner = T1GraphRunner(self)
        self.T21_graph_runner = T21GraphRunner(self)
        self.T2_graph_runner = T2GraphRunner(self)
        self.T4_graph_runner = T4GraphRunner(self)
        self.Tvv_graph_runner = TvvGraphRunner(self)

        self.num_samples = lambda nrv: math.comb(nrv, 2)
        self.batch_size = lambda nrv: int(nrv*0.1)

        self.device = hmc.device
        self.dtype = hmc.dtype
        self.cdtype = hmc.cdtype

        # init
        if hmc.precon_csr is None and hmc.dtau <= 0.1 and precon_on:
            hmc.reset_precon()

        # Cache
        # self.GD0_G0D = None
        # self.GD0 = None

    def initialize(self):
        # Tunable parameters
        # Batch processing to avoid OOM
        batch_size = self.batch_size(self.Nrv)
        inner_batch_size = batch_size  # int(0.1*Nrv)
        num_inner_loops = self.num_inner_loops # 200
        outer_stride = inner_batch_size * num_inner_loops

        params = Params()
        params.num_inner_loops = num_inner_loops
        params.inner_batch_size = inner_batch_size
        params.total_num_samples = self.num_samples(self.Nrv)
        params.outer_stride = outer_stride
        params.Ly = self.Ly
        params.Lx = self.Lx
        self.params = params

        self.indices_r2 = torch.combinations(torch.arange(self.Nrv, device=self.device), r=2, with_replacement=False)

    def init_cuda_graph(self):
        hmc = self.hmc_sampler
        print(f"Nrv: {self.Nrv}")
        print(f"max_iter_se: {hmc._MAX_ITERS_TO_CAPTURE[0]}")
        # Capture
        if self.cuda_graph_se and capture_fermion_obsr:
            print("Initializing CUDA graph for get_fermion_obsr.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init se_graph: {d_mem_str}")
            dummy_eta = torch.zeros((self.Nrv, self.Ltau * self.Vs), device=hmc.device, dtype=hmc.cdtype)
            dummy_bosons = torch.zeros((hmc.bs, 2, self.Lx, self.Ly, self.Ltau), device=hmc.device, dtype=hmc.dtype)
            dummy_indices = torch.zeros((self.num_samples(self.Nrv), 4), device=hmc.device, dtype=torch.int64)
            dummy_indices_r2 = torch.zeros((self.num_samples(self.Nrv), 2), device=hmc.device, dtype=torch.int64)
            self.graph_memory_pool = self.graph_runner.capture(
                                        dummy_bosons, 
                                        dummy_eta, 
                                        dummy_indices,
                                        dummy_indices_r2,
                                        max_iter_se=hmc._MAX_ITERS_TO_CAPTURE[0],
                                        graph_memory_pool=self.graph_memory_pool)
            print(f"get_fermion_obsr CUDA graph initialization complete")
            print('')

        if self.cuda_graph_se:
            # G_eta_graph_runner
            print("Initializing G_eta_graph_runner.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init G_eta_graph_runner: {d_mem_str}")
            self.graph_memory_pool = self.G_eta_graph_runner.capture(
                max_iter_se=hmc._MAX_ITERS_TO_CAPTURE[0],
                graph_memory_pool=self.graph_memory_pool)
            print(f"G_eta_graph_runner initialization complete")
            print('')

            # # L0_graph_runner
            # print("Initializing L0_graph_runner.........")
            # d_mem_str, d_mem2 = device_mem()
            # print(f"Before init L0_graph_runner: {d_mem_str}")
            # self.graph_memory_pool = self.L0_graph_runner.capture(
            #     params=self.params,
            #     graph_memory_pool=self.graph_memory_pool)
            # print(f"L0_graph_runner initialization complete")
            # print('')

            # spsm_graph_runner
            print("Initializing spsm_graph_runner.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init spsm_graph_runner: {d_mem_str}")
            self.graph_memory_pool = self.spsm_graph_runner.capture(
                graph_memory_pool=self.graph_memory_pool)
            print(f"spsm_graph_runner initialization complete")
            print('')

            # T1_graph_runner
            print("Initializing T1_graph_runner.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init T1_graph_runner: {d_mem_str}")
            self.graph_memory_pool = self.T1_graph_runner.capture(
                graph_memory_pool=self.graph_memory_pool)
            print(f"T1_graph_runner initialization complete")
            print('')

            # T2_graph_runner
            print("Initializing T2_graph_runner.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init T2_graph_runner: {d_mem_str}")
            self.graph_memory_pool = self.T21_graph_runner.capture(
                graph_memory_pool=self.graph_memory_pool)
            print(f"T2_graph_runner initialization complete")
            print('')

            # T3_graph_runner
            print("Initializing T3_graph_runner.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init T3_graph_runner: {d_mem_str}")
            self.graph_memory_pool = self.T2_graph_runner.capture(
                graph_memory_pool=self.graph_memory_pool)
            print(f"T3_graph_runner initialization complete")
            print('')

            # T4_graph_runner
            print("Initializing T4_graph_runner.........")
            d_mem_str, d_mem2 = device_mem()
            print(f"Before init T4_graph_runner: {d_mem_str}")
            self.graph_memory_pool = self.T4_graph_runner.capture(
                graph_memory_pool=self.graph_memory_pool)
            print(f"T4_graph_runner initialization complete")
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

    @staticmethod
    def reorder_fft_grid2_inverse(tensor2d, dims=(-2, -1)):
        """Reverse the effect of reorder_fft_grid2, restoring FFT-style order."""
        Ny, Nx = tensor2d.shape[dims[0]], tensor2d.shape[dims[1]]
        return torch.roll(tensor2d, shifts=(-Ny // 2, -Nx // 2), dims=dims)

    def test_orthogonality(self, rand_vec):
        """
        Test the orthogonality of the random vectors
        """
        Nrv = rand_vec.shape[0]
        Ltau = rand_vec.shape[1]
        rand_vec = rand_vec.reshape(Nrv*Ltau, -1)  # [Nrv, Ltau, Ly * Lx]
        
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

    
    def test_ortho_two_identities(self, eta):
        Nrv = eta.shape[0]
        Ltau = eta.shape[1]
        eta = eta.view(Nrv, Ltau, self.Ly * self.Lx) # [Nrv, Ltau, Ly * Lx]
        eta_conj = eta.conj()                        # [Nrv, Ltau, Ly * Lx]
        sa, sb = self.indices_r2[:300, 0], self.indices_r2[:300, 1] 

        eta_a = eta[sa]
        eta_b = eta[sb]
        eta_conj_a = eta_conj[sa]
        eta_conj_b = eta_conj[sb]
        external_prod = torch.einsum('ati,atj,atm,atn->atijmn', eta_conj_a, eta_a, eta_conj_b, eta_b).mean(dim=(0, 1)) # [vs, vs, vs, vs]

        # external_prod: [Ltau, Ly*Lx, Ly*Lx, Ly*Lx, Ly*Lx]
        # Check that external_prod is close to delta_ij * delta_mn
        N = self.Ly * self.Lx
        delta_ij = torch.eye(N, device=eta.device)
        delta_mn = torch.eye(N, device=eta.device)
        delta_prod = torch.einsum('ij,mn->ijmn', delta_ij, delta_mn)  # [N, N, N, N]
        try:
            torch.testing.assert_close(external_prod.real, delta_prod, rtol=1e-2, atol=1e-2)
        except Exception as e:
            print("Error info:", e)


        # Filter out entries of external_product that are smaller than 1e-3
        mask = external_prod.abs() < 1e-3
        external_prod[mask] = 0

        # Output the check results of the two identities
        print("external_prod.real[..., 0, 0]:\n", external_prod.real[..., 0, 0])
        print("external_prod.imag[..., 0, 0]:\n", external_prod.imag[..., 0, 0])
        print("external_prod.real[0, 0, ...]:\n", external_prod.real[0, 0, ...])
        print("external_prod.imag[0, 0, ...]:\n", external_prod.imag[0, 0, ...])
        print("external_prod shape:", external_prod.shape)
        diff = (external_prod.real - delta_prod).abs()
        print("Max absolute difference from two-identity orthogonality (atol):", diff.max().item())
        atol = diff.max()
        return atol
    
    def test_ortho_four_identities(self, eta):
        Nrv = eta.shape[0]
        Ltau = eta.shape[1]
        eta = eta.view(Nrv, Ltau, self.Ly * self.Lx) # [Nrv, Ltau, Ly * Lx]
        eta_conj = eta.conj()                        # [Nrv, Ltau, Ly * Lx]
        sa, sb, sc, sd = self.indices[:1000, 0], self.indices[:1000, 1] 

        eta_a = eta[sa]
        eta_b = eta[sb]
        eta_conj_a = eta_conj[sa]
        eta_conj_b = eta_conj[sb]
        external_prod = torch.einsum('ati,atj,atm,atn->atijmn', eta_conj_a, eta_a, eta_conj_b, eta_b).mean(dim=(0, 1)) # [vs, vs, vs, vs]

        # external_prod: [Ltau, Ly*Lx, Ly*Lx, Ly*Lx, Ly*Lx]
        # Check that external_prod is close to delta_ij * delta_mn
        N = self.Ly * self.Lx
        delta_ij = torch.eye(N, device=eta.device)
        delta_mn = torch.eye(N, device=eta.device)
        delta_prod = torch.einsum('ij,mn->ijmn', delta_ij, delta_mn)  # [N, N, N, N]
        try:
            torch.testing.assert_close(external_prod.real, delta_prod, rtol=1e-2, atol=1e-2)
        except Exception as e:
            print("Error info:", e)

        # Filter out entries of external_product that are smaller than 1e-3
        mask = external_prod.abs() < 1e-3
        external_prod[mask] = 0

        # Output the check results of the two identities
        print("external_prod.real[..., 0, 0]:\n", external_prod.real[..., 0, 0])
        print("external_prod.imag[..., 0, 0]:\n", external_prod.imag[..., 0, 0])
        print("external_prod.real[0, 0, ...]:\n", external_prod.real[0, 0, ...])
        print("external_prod.imag[0, 0, ...]:\n", external_prod.imag[0, 0, ...])
        print("external_prod shape:", external_prod.shape)
        diff = (external_prod.real - delta_prod).abs()
        print("Max absolute difference from two-identity orthogonality (atol):", diff.max().item())
        atol = diff.max()
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
        self.eta = eta  # [Nrv, Ltau * Ly * Lx]

        if self.cuda_graph_se:
            G_eta = self.G_eta_graph_runner(boson, eta)
        else:
            G_eta = self.set_eta_G_eta_inner(boson, eta)  # [Nrv, Ltau * Ly * Lx]

        # torch.testing.assert_close(G_eta, G_eta_ref, rtol=1e-3, atol=1e-3)

        self.G_eta = G_eta

    def set_eta_G_eta_inner(self, boson, eta):
        self.hmc_sampler.bs, bs = self.Nrv, self.hmc_sampler.bs
        boson = boson.permute([0, 4, 3, 2, 1]).reshape(1, -1).repeat(self.Nrv, 1)  # [Nrv, Ltau * Ly * Lx]
        psudo_fermion = _C.mh_vec(boson, eta, self.Lx, self.dtau, *BLOCK_SIZE)  # [Nrv, Ltau * Ly * Lx]
        G_eta, cnt, err = self.hmc_sampler.Ot_inv_psi_fast(psudo_fermion, boson.view(self.Nrv, self.Ltau, -1), None)  # [Nrv, Ltau * Ly * Lx]
        self.hmc_sampler.bs = bs
        return G_eta
  
    def test_fft_negate_k3(self):
        device = self.device
        # Create 3D frequency grid for (2*Ltau, Ly, Lx)
        k_tau = torch.fft.fftfreq(2 * self.Ltau, device=self.device)
        k_y = torch.fft.fftfreq(self.Ly, device=self.device)
        k_x = torch.fft.fftfreq(self.Lx, device=self.device)
        ks = torch.stack(torch.meshgrid(k_tau, k_y, k_x, indexing='ij'), dim=-1)  # shape: (2*Ltau, Ly, Lx, 3)
        ks_neg = self.fft_negate_k3(ks.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)

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
    
    def GD0_func(self, eta, G_eta):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

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

    def G_eqt_se(self):
        Ltau, Ly, Lx = self.Ltau, self.Ly, self.Lx
        eta_conj = self.eta.view(-1, Ltau, Ly*Lx).conj()  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta.view(-1, Ltau, Ly*Lx)  # [Nrv, Ltau * Ly * Lx]

        # G_ij = G_eta_i * eta_conj_j
        G_eqt_se = torch.einsum('ati,atj->atij', G_eta, eta_conj).mean(dim=0)  # [Ltau, vs, vs]
        return G_eqt_se # [Ltau, vs, vs]

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

    # -------- Ext methods --------
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

    # -------- Batched methods --------
    def G_delta_0_G_0_delta_ext_batch(self, a_xi=0, a_G_xi=0, b_xi=0, b_G_xi=0):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj().view(-1, 2 * self.Ltau, self.Ly, self.Lx)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1).view(-1, 2 * self.Ltau, self.Ly, self.Lx)

        # Get all unique pairs (s, s_prime) with s < s_prime
        Nrv = eta_ext_conj.shape[0]

        # Use torch.combinations to get all unique pairs (s, s_prime) with s < s_prime
        s, s_prime = self.indices_r2[:, 0], self.indices_r2[:, 1]

        # a = eta_ext_conj[s] * G_eta_ext[s_prime]
        # b = eta_ext_conj[s_prime] * G_eta_ext[s]
        #
        # a = a.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]
        # b = b.view(-1, 2*self.Ltau, self.Ly, self.Lx)  # [N, 2Ltau, Ly, Lx]
        #
        # a_F_neg_k = torch.fft.ifftn(a, (2*self.Ltau, self.Ly, self.Lx), norm="backward")
        # b_F = torch.fft.fftn(b, (2*self.Ltau, self.Ly, self.Lx), norm="forward")
        # G_delta_0_G_delta_0 = torch.fft.ifftn(a_F_neg_k * b_F, (2*self.Ltau, self.Ly, self.Lx), norm="forward").mean(dim=0)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints
        # if a_xi == a_G_xi == b_xi == b_G_xi ==0:
        #     print(f"Batch size for G_delta_0_G_0_delta_ext: {batch_size}")
        
        # batch_size = len(s)
        # num_loop = 10
        # batch_size = total_pairs // num_loop
        G_delta_0_G_delta_0_sum = torch.zeros((2 * self.Ltau, self.Ly, self.Lx),
                                              dtype=eta_ext_conj.dtype, device=eta.device)
        total_pairs = len(s)

        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            # a = eta_ext_conj[s_batch] * G_eta_ext[s_prime_batch]
            # b = eta_ext_conj[s_prime_batch] * G_eta_ext[s_batch]
            a = torch.roll(eta_ext_conj[s_batch], shifts=a_xi, dims=-1) * torch.roll(G_eta_ext[s_prime_batch], shifts=a_G_xi, dims=-1)
            b = torch.roll(eta_ext_conj[s_prime_batch], shifts=b_xi, dims=-1) * torch.roll(G_eta_ext[s_batch], shifts=b_G_xi, dims=-1)

            a_F_neg_k = torch.fft.ifftn(a, (2 * self.Ltau, self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_sum += batch_result.sum(dim=0)

        G_delta_0_G_delta_0 = G_delta_0_G_delta_0_sum / total_pairs

        return G_delta_0_G_delta_0.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]

    def GD0_G0D_func(self, eta, G_eta, a_xi=0, a_G_xi=0, b_xi=0, b_G_xi=0):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj().view(-1, 2 * self.Ltau, self.Ly, self.Lx)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1).view(-1, 2 * self.Ltau, self.Ly, self.Lx)

        # Get all unique pairs (s, s_prime) with s < s_prime
        Nrv = eta_ext_conj.shape[0]

        # Use torch.combinations to get all unique pairs (s, s_prime) with s < s_prime
        s, s_prime = self.indices_r2[:, 0], self.indices_r2[:, 1]

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))

        G_delta_0_G_delta_0_sum = torch.zeros((2 * self.Ltau, self.Ly, self.Lx),
                                              dtype=eta_ext_conj.dtype, device=eta.device)
        total_pairs = len(s)
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            a = torch.roll(eta_ext_conj[s_batch], shifts=a_xi, dims=-1) * torch.roll(G_eta_ext[s_prime_batch], shifts=a_G_xi, dims=-1)
            b = torch.roll(eta_ext_conj[s_prime_batch], shifts=b_xi, dims=-1) * torch.roll(G_eta_ext[s_batch], shifts=b_G_xi, dims=-1)

            a_F_neg_k = torch.fft.ifftn(a, (2 * self.Ltau, self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_sum += batch_result.sum(dim=0)

        G_delta_0_G_delta_0 = G_delta_0_G_delta_0_sum / total_pairs

        return G_delta_0_G_delta_0.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]

    def L8(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = eta_conj[sa_batch] * \
                G_eta[sb_batch] * \
                torch.roll(eta_conj[sc_batch], shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch], shifts=-1, dims=-1)
            
            b = G_eta[sa_batch] * \
                torch.roll(eta_conj[sb_batch], shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch], shifts=-1, dims=-1) * \
                eta_conj[sd_batch]
            
            c = G_eta[sb_batch] *\
                    torch.roll(eta_conj[sb_batch], shifts=-1, dims=-1) * \
                    torch.roll(G_eta[sc_batch], shifts=-1, dims=-1) * \
                    torch.roll(eta_conj[sc_batch], shifts=-1, dims=-1) * \
                    torch.roll(G_eta[sd_batch], shifts=-1, dims=-1) * \
                    eta_conj[sd_batch]

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean += c.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean = - G_delta_0_G_delta_0_mean
        G_delta_0_G_delta_0_mean[0, 0] += G_res_mean[0]

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
    
    def L7(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = eta_conj[sa_batch] * \
                G_eta[sb_batch] * \
                torch.roll(G_eta[sc_batch], shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch], shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sa_batch], shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sb_batch], shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch], shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch], shifts=0, dims=-1)
            
            c = torch.roll(G_eta[sb_batch], shifts=-1, dims=-1) *\
                torch.roll(eta_conj[sb_batch], shifts=0, dims=-1) * \
                torch.roll(G_eta[sc_batch], shifts=-2, dims=-1) * \
                torch.roll(eta_conj[sc_batch], shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch], shifts=0, dims=-1) * \
                torch.roll(eta_conj[sd_batch], shifts=-2, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean += c.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean = - G_delta_0_G_delta_0_mean
        G_delta_0_G_delta_0_mean[0, -1] += G_res_mean[0]

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L6(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(G_eta[sa_batch], shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sa_batch], shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch], shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch], shifts=-1, dims=-1)
            
            b = torch.roll(eta_conj[sb_batch], shifts=0, dims=-1) * \
                torch.roll(G_eta[sc_batch], shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch], shifts=0, dims=-1) * \
                torch.roll(eta_conj[sd_batch], shifts=-1, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean = - G_delta_0_G_delta_0_mean
        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L5(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sa_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=0, dims=-1)
            
            c = torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) *\
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=0, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean += c.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean = - G_delta_0_G_delta_0_mean
        G_delta_0_G_delta_0_mean[0, 0] += G_res_mean[0]

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
    
    def L4(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(G_eta[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1)

            b = torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=0, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=0, dims=-1)
    
            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean = - G_delta_0_G_delta_0_mean

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L3(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)
            
            c = torch.roll(G_eta[sb_batch],     shifts=-1, dims=-1) *\
                torch.roll(eta_conj[sb_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-2, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean += c.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean = - G_delta_0_G_delta_0_mean
        G_delta_0_G_delta_0_mean[0, -1] += G_res_mean[0]

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
 
    def L2_half_n_choose_4(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0] // 2
        eta_conj1 = eta_conj[:Nrv]
        G_eta1 = G_eta[:Nrv]

        eta_conj2 = eta_conj[Nrv:]
        G_eta2 = G_eta[Nrv:]

        # Get all unique tuples (sa, sb) with sa < sb
        sa, sb = self.indices_r2[:, 0], self.indices_r2[:, 1]

        num_samples = self.num_samples(Nrv)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean1 = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean2 = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]

            a = torch.roll(eta_conj1[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta1[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj2[sa_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta2[sb_batch],     shifts=-1, dims=-1)
            
            b = torch.roll(G_eta1[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj1[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta2[sa_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj2[sb_batch],  shifts=-0, dims=-1)
            
            c1 = torch.roll(G_eta1[sb_batch],    shifts=-1, dims=-1) *\
                torch.roll(eta_conj1[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta2[sa_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj2[sa_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta2[sb_batch],     shifts=-2, dims=-1) * \
                torch.roll(eta_conj2[sb_batch],  shifts=-0, dims=-1)
            
            c2 = torch.roll(G_eta1[sa_batch],    shifts=-2, dims=-1) *\
                torch.roll(eta_conj1[sa_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta1[sb_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj1[sb_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta2[sb_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj2[sb_batch],  shifts=-1, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean1 += c1.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples
            G_res_mean2 += c2.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean[0, -1] -= G_res_mean1[0]
        G_delta_0_G_delta_0_mean[0, 1] -= G_res_mean2[0]
        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L2(self):
        eta = self.eta[:self.Nrv]  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta[:self.Nrv]  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean1 = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean2 = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-0, dims=-1)
            
            c1 = torch.roll(G_eta[sb_batch],    shifts=-1, dims=-1) *\
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-2, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-0, dims=-1)
            
            c2 = torch.roll(G_eta[sa_batch],    shifts=-2, dims=-1) *\
                torch.roll(eta_conj[sa_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean1 += c1.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples
            G_res_mean2 += c2.mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean[0, -1] -= G_res_mean1[0]
        G_delta_0_G_delta_0_mean[0, 1] -= G_res_mean2[0]
        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L2_groundtruth(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints
        N = self.Lx * self.Ly

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean1 = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean2 = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-0, dims=-1)

            ca1 =   torch.roll(G_eta[sb_batch],     shifts=-0, dims=-1) * \
                    torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                    torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1)

            cb1 =   torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                    torch.roll(G_eta[sc_batch],     shifts=-0, dims=-1) * \
                    torch.roll(eta_conj[sd_batch],  shifts=-0, dims=-1)
            
            ca2 =   torch.roll(eta_conj[sa_batch],  shifts=-0, dims=-1) * \
                    torch.roll(G_eta[sb_batch],     shifts=-0, dims=-1) * \
                    torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) 
            cb2 =   torch.roll(G_eta[sa_batch],     shifts=-1, dims=-1) * \
                    torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1) * \
                    torch.roll(eta_conj[sd_batch],  shifts=-0, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            ca1_F_neg_k = torch.fft.ifftn(ca1, (self.Ly, self.Lx), norm="backward")
            cb1_F = torch.fft.fftn(cb1, (self.Ly, self.Lx), norm="forward")
            c1 = torch.fft.ifftn(ca1_F_neg_k * cb1_F, (self.Ly, self.Lx), norm="forward")

            ca2_F_neg_k = torch.fft.ifftn(ca2, (self.Ly, self.Lx), norm="backward")
            cb2_F = torch.fft.fftn(cb2, (self.Ly, self.Lx), norm="forward")
            c2 = torch.fft.ifftn(ca2_F_neg_k * cb2_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples # [Ly, Lx]
            G_res_mean1 += c1.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples    # [Ly, Lx]
            G_res_mean2 += c2.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples    # [Ly, Lx]

        G_delta_0_G_delta_0_mean[0, -1] -= G_res_mean1[0, -1]
        G_delta_0_G_delta_0_mean[0, 1] -= G_res_mean2[0, 1]
        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L1(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(sa), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # indices = perm[start_idx:end_idx]
            sa_batch = sa[start_idx:end_idx]
            sb_batch = sb[start_idx:end_idx]
            sc_batch = sc[start_idx:end_idx]
            sd_batch = sd[start_idx:end_idx]

            a = torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sa_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)
            
            c1 = torch.roll(G_eta[sa_batch],    shifts=-0, dims=-1) *\
                torch.roll(eta_conj[sa_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)
            
            c2 = torch.roll(G_eta[sb_batch],    shifts=-0, dims=-1) *\
                torch.roll(eta_conj[sb_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sc_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)
            
            c3 = torch.roll(G_eta[sb_batch],    shifts=-0, dims=-1) *\
                torch.roll(eta_conj[sb_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1) 

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_res_mean += (-c1 - c2 + c3).mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples

        G_delta_0_G_delta_0_mean[0, 0] += G_res_mean[0]

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    def L0(self):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        # Get all unique quadruples (sa, sb, sc, sd) with sa < sb < sc < sd
        Nrv = eta_conj.shape[0]
        # idx = torch.combinations(torch.arange(Nrv, device=eta.device), r=4, with_replacement=False)
        # sa, sb, sc, sd = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        sa, sb, sc, sd = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2], self.indices[:, 3]
        
        total_num_samples = self.num_samples(Nrv)
        # perm = torch.randperm(len(sa), device=eta.device)

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)

        # # Batch processing to avoid OOM
        # batch_size = min(len(sa), self.batch_size(Nrv))
        # # Tunable parameters
        # inner_batch_size = batch_size  # int(0.1*Nrv)
        # num_inner_loops = self.num_inner_loops # 200

        # outer_stride = inner_batch_size * num_inner_loops

        outer_stride = self.params.outer_stride
        num_outer_loops = math.ceil(total_num_samples / outer_stride)

        for chunk_idx in range(num_outer_loops):
            sa_chunk = sa[chunk_idx * outer_stride: (chunk_idx + 1) * outer_stride]
            sb_chunk = sb[chunk_idx * outer_stride: (chunk_idx + 1) * outer_stride]
            sc_chunk = sc[chunk_idx * outer_stride: (chunk_idx + 1) * outer_stride]
            sd_chunk = sd[chunk_idx * outer_stride: (chunk_idx + 1) * outer_stride]

            # Pad sa_chunk, sb_chunk, sc_chunk, sd_chunk if they are shorter than outer_stride
            if len(sa_chunk) < outer_stride:
                padding_length = outer_stride - len(sa_chunk)
                sa_chunk = F.pad(sa_chunk, (0, padding_length), value=0)
                sb_chunk = F.pad(sb_chunk, (0, padding_length), value=0)
                sc_chunk = F.pad(sc_chunk, (0, padding_length), value=0)
                sd_chunk = F.pad(sd_chunk, (0, padding_length), value=0)

            if self.cuda_graph_se:
                diff_G_delta_0_G_delta_0_mean = self.L0_graph_runner(sa_chunk, sb_chunk, sc_chunk, sd_chunk, G_eta, eta_conj)
            else:
                diff_G_delta_0_G_delta_0_mean = self.L0_inner(sa_chunk, sb_chunk, sc_chunk, sd_chunk, G_eta, eta_conj, self.params)

            # torch.testing.assert_close(diff_G_delta_0_G_delta_0_mean_ref, diff_G_delta_0_G_delta_0_mean, rtol=1e-5, atol=1e-5)

            G_delta_0_G_delta_0_mean += diff_G_delta_0_G_delta_0_mean

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]

    @staticmethod
    def L0_inner(sa_chunk, sb_chunk, sc_chunk, sd_chunk, G_eta, eta_conj, params):
        # TODO: a mask is needed when sa_chunk etc are padded.
        num_inner_loops = params.num_inner_loops
        inner_batch_size = params.inner_batch_size
        outer_stride = params.outer_stride

        G_delta_0_G_delta_0_mean = torch.zeros((params.Ly, params.Lx), dtype=G_eta.dtype, device=G_eta.device)

        for inner_loop in range(num_inner_loops):
            start_idx = inner_loop * inner_batch_size
            end_idx = min(start_idx + inner_batch_size, outer_stride)
            # if start_idx >= outer_stride: break
            sa_batch = sa_chunk[start_idx:end_idx]
            sb_batch = sb_chunk[start_idx:end_idx]
            sc_batch = sc_chunk[start_idx:end_idx]
            sd_batch = sd_chunk[start_idx:end_idx]

            a = torch.roll(G_eta[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sb_batch],  shifts=-1, dims=-1)
            
            b = torch.roll(G_eta[sc_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj[sc_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta[sd_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj[sd_batch],  shifts=-1, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (params.Ly, params.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (params.Ly, params.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (params.Ly, params.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / params.total_num_samples

        return G_delta_0_G_delta_0_mean

    def T1(self, eta, G_eta, boson):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]
        # boson: [1, 2, Lx, Ly, Ltau]
        boson = boson.permute(0, 1, 4, 3, 2) # [1, 2, Ltau, Ly, Lx]

        # Use torch.combinations to get all unique pairs (s, s_prime) with s < s_prime
        s, s_prime = self.indices_r2[:, 0], self.indices_r2[:, 1]
        
        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0]
        num_samples = self.num_samples(Nrv)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_diag_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            a = torch.roll(eta_conj[s_batch],       shifts=-0, dims=-1) * \
                torch.roll(G_eta[s_prime_batch],    shifts=-1, dims=-1) * \
                torch.exp(1j * boson[0, 0, :, :, :])

            b = torch.roll(G_eta[s_batch],          shifts=-1, dims=-1) * \
                torch.roll(eta_conj[s_prime_batch], shifts=-0, dims=-1) * \
                torch.exp(1j * boson[0, 0, :, :, :])

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += - batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
        
        G_diag_mean = torch.roll(G_eta,     shifts=-1, dims=-1) * \
                      torch.roll(eta_conj,  shifts=+1, dims=-1) * \
                      torch.exp(1j * boson[0, 0, :, :, :])      * \
                      torch.exp(1j * torch.roll(boson, shifts=+1, dims=-1)[0, 0, :, :, :])

        G_diag_mean = G_diag_mean.mean(dim=(0, 1, 2, 3))
        G_delta_0_G_delta_0_mean[0, -1] += G_diag_mean

        # G_diag_mean += (-c1 - c2 + c3).mean(dim=(0, 1, 2, 3)) * (end_idx - start_idx) / num_samples
        # G_delta_0_G_delta_0_mean[0, 0] += G_res_mean[0]
        
        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
    
    def T21(self, eta, G_eta, boson):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]
        # boson: [1, 2, Lx, Ly, Ltau]
        boson = boson.permute(0, 1, 4, 3, 2) # [1, 2, Ltau, Ly, Lx]

        # Use torch.combinations to get all unique pairs (s, s_prime) with s < s_prime
        s, s_prime = self.indices_r2[:, 0], self.indices_r2[:, 1]
        
        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0]
        num_samples = self.num_samples(Nrv)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_diag_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            a = (
                torch.roll(eta_conj[s_batch],       shifts=-1, dims=-1) * \
                torch.roll(G_eta[s_prime_batch],    shifts=-0, dims=-1) 
                + \
                torch.roll(eta_conj[s_prime_batch], shifts=-1, dims=-1) * \
                torch.roll(G_eta[s_batch],          shifts=-0, dims=-1)
            ) * torch.exp(-1j * boson[0, 0, :, :, :]) / 2

            b = (
                torch.roll(G_eta[s_batch],          shifts=-1, dims=-1) * \
                torch.roll(eta_conj[s_prime_batch], shifts=-0, dims=-1) 
                + \
                torch.roll(G_eta[s_prime_batch],    shifts=-1, dims=-1) * \
                torch.roll(eta_conj[s_batch],       shifts=-0, dims=-1) 
            ) * torch.exp(1j * boson[0, 0, :, :, :]) / 2

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += - batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
        
        G_diag_mean = torch.roll(G_eta,     shifts=-0, dims=-1) * \
                      torch.roll(eta_conj,  shifts=-0, dims=-1)

        G_diag_mean = G_diag_mean.mean(dim=(0, 1, 2, 3))
        G_delta_0_G_delta_0_mean[0, 0] += G_diag_mean

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
    
    def T2(self, eta, G_eta, boson):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]
        # boson: [1, 2, Lx, Ly, Ltau]
        boson = boson.permute(0, 1, 4, 3, 2) # [1, 2, Ltau, Ly, Lx]

        # Use torch.combinations to get all unique pairs (s, s_prime) with s < s_prime
        s, s_prime = self.indices_r2[:, 0], self.indices_r2[:, 1]
        
        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0]
        num_samples = self.num_samples(Nrv)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_diag_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            a = (
                torch.roll(eta_conj[s_batch],       shifts=-0, dims=-1) * \
                torch.roll(G_eta[s_prime_batch],    shifts=-1, dims=-1)
                + \
                torch.roll(eta_conj[s_prime_batch], shifts=-0, dims=-1) * \
                torch.roll(G_eta[s_batch],          shifts=-1, dims=-1) 
                ) * torch.exp(1j * boson[0, 0, :, :, :]) / 2

            b = (
                torch.roll(G_eta[s_batch],          shifts=-0, dims=-1) * \
                torch.roll(eta_conj[s_prime_batch], shifts=-1, dims=-1)
                + \
                torch.roll(G_eta[s_prime_batch],          shifts=-0, dims=-1) * \
                torch.roll(eta_conj[s_batch], shifts=-1, dims=-1)
                ) * torch.exp(-1j * boson[0, 0, :, :, :]) / 2

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += - batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
        
        G_diag_mean = torch.roll(G_eta,     shifts=0, dims=-1) * \
                      torch.roll(eta_conj,  shifts=0, dims=-1)

        G_diag_mean = G_diag_mean.mean(dim=(0, 1, 2, 3))
        G_delta_0_G_delta_0_mean[0, 0] += G_diag_mean

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
    
    def T4(self, eta, G_eta, boson):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]
        # boson: [1, 2, Lx, Ly, Ltau]
        boson = boson.permute(0, 1, 4, 3, 2) # [1, 2, Ltau, Ly, Lx]

        # Use torch.combinations to get all unique pairs (s, s_prime) with s < s_prime
        s, s_prime = self.indices_r2[:, 0], self.indices_r2[:, 1]
        
        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0]
        num_samples = self.num_samples(Nrv)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_diag_mean = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            a = torch.roll(eta_conj[s_batch],       shifts=-1, dims=-1) * \
                torch.roll(G_eta[s_prime_batch],    shifts=0 , dims=-1) * \
                torch.exp(-1j * boson[0, 0, :, :, :])

            b = torch.roll(G_eta[s_batch],          shifts=0 , dims=-1) * \
                torch.roll(eta_conj[s_prime_batch], shifts=-1, dims=-1) * \
                torch.exp(-1j * boson[0, 0, :, :, :])

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_mean += - batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
        
        G_diag_mean = torch.roll(G_eta,     shifts=0, dims=-1) * \
                      torch.roll(eta_conj,  shifts=-2, dims=-1) * \
                      torch.exp(-1j * boson[0, 0, :, :, :])      * \
                      torch.exp(-1j * torch.roll(boson, shifts=-1, dims=-1)[0, 0, :, :, :])

        G_diag_mean = G_diag_mean.mean(dim=(0, 1, 2, 3))
        G_delta_0_G_delta_0_mean[0, 1] += G_diag_mean

        return G_delta_0_G_delta_0_mean  # [Ly, Lx]
    
    def Tvv(self, eta, G_eta, boson):
        # eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        # G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]
        # boson: [1, 2, Lx, Ly, Ltau]
        boson = boson.permute(0, 1, 4, 3, 2) # [1, 2, Ltau, Ly, Lx]
        
        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0]
        num_samples = Nrv

        # Batch processing to avoid OOM
        batch_size = min(self.batch_size(Nrv), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            s_batch = torch.range(start_idx, end_idx, dtype=torch.int64, device=boson.device)

            v = torch.roll(G_eta[s_batch],      shifts=-1, dims=-1) * \
                torch.roll(eta_conj[s_batch],   shifts=0 , dims=-1) * \
                torch.exp(1j * boson[0, 0, :, :, :]) + \
                torch.roll(G_eta[s_batch],      shifts=0, dims=-1) * \
                torch.roll(eta_conj[s_batch],   shifts=-1, dims=-1) * \
                torch.exp(-1j * boson[0, 0, :, :, :])
            
            a = v
            b = v
            
            # FFT: \sum_i ai * b_{i+delta}
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward") # [Nrv, Ltau, Ly, Lx]

            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples
            G_delta_0_mean += v.mean(dim=(0, 1)) * (end_idx - start_idx) / num_samples

        return G_delta_0_G_delta_0_mean, G_delta_0_mean  # [Ly, Lx], [Ly, Lx]
    

    def G_delta_delta_G_0_0_ext_batch(self, a_xi=0, a_G_xi=0, b_xi=0, b_G_xi=0):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj().view(-1, 2 * self.Ltau, self.Ly, self.Lx)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1).view(-1, 2 * self.Ltau, self.Ly, self.Lx)

        # Get all unique pairs (s, s_prime) with s < s_prime
        Nrv = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(Nrv, Nrv, offset=1, device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_delta_G_0_0_sum = torch.zeros((2 * self.Ltau, self.Ly, self.Lx),
                                              dtype=eta_ext_conj.dtype, device=eta.device)
        total_pairs = len(s)
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]
            
            # a = eta_ext_conj[s_batch] * G_eta_ext[s_batch]
            # b = eta_ext_conj[s_prime_batch] * G_eta_ext[s_prime_batch]
            a = torch.roll(eta_ext_conj[s_batch], shifts=a_xi, dims=-1) * torch.roll(G_eta_ext[s_batch], shifts=a_G_xi, dims=-1)
            b = torch.roll(eta_ext_conj[s_prime_batch], shifts=b_xi, dims=-1) * torch.roll(G_eta_ext[s_prime_batch], shifts=b_G_xi, dims=-1)

            a_F_neg_k = torch.fft.ifftn(a, (2 * self.Ltau, self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")

            G_delta_delta_G_0_0_sum += batch_result.sum(dim=0)

        G_delta_delta_G_0_0 = G_delta_delta_G_0_0_sum / total_pairs

        return G_delta_delta_G_0_0.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]

    def G_0_delta_G_0_delta_ext_batch(self, a_G_xi=0, a_G_xi_prime=0, b_xi=0, b_xi_prime=0):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj().view(-1, 2 * self.Ltau, self.Ly, self.Lx)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1).view(-1, 2 * self.Ltau, self.Ly, self.Lx)

        # Get all unique pairs (s, s_prime) with s < s_prime
        Nrv = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(Nrv, Nrv, offset=1, device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_0_delta_G_0_delta_sum = torch.zeros((2 * self.Ltau, self.Ly, self.Lx),
                                              dtype=eta_ext_conj.dtype, device=eta.device)
        total_pairs = len(s)
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]

            # a = G_eta_ext[s_batch] * G_eta_ext[s_prime_batch]
            # b = eta_ext_conj[s_batch] * eta_ext_conj[s_prime_batch]
            a = torch.roll(G_eta_ext[s_batch], shifts=a_G_xi, dims=-1) * torch.roll(G_eta_ext[s_prime_batch], shifts=a_G_xi_prime, dims=-1)
            b = torch.roll(eta_ext_conj[s_batch], shifts=b_xi, dims=-1) * torch.roll(eta_ext_conj[s_prime_batch], shifts=b_xi_prime, dims=-1)

            a_F_neg_k = torch.fft.ifftn(a, (2 * self.Ltau, self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")

            G_0_delta_G_0_delta_sum += batch_result.sum(dim=0)

        G_0_delta_G_0_delta = G_0_delta_G_0_delta_sum / total_pairs

        return G_0_delta_G_0_delta.view(2*self.Ltau, self.Ly, self.Lx)[:self.Ltau]

    def G_delta_0_G_delta_0_ext_batch(self, a_xi=0, a_xi_prime=0, b_G_xi=0, b_G_xi_prime=0):
        eta = self.eta  # [Nrv, Ltau * Ly * Lx]
        G_eta = self.G_eta  # [Nrv, Ltau * Ly * Lx]

        eta_ext_conj = torch.cat([eta, -eta], dim=1).conj().view(-1, 2 * self.Ltau, self.Ly, self.Lx)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1).view(-1, 2 * self.Ltau, self.Ly, self.Lx)

        # Get all unique pairs (s, s_prime) with s < s_prime
        Nrv = eta_ext_conj.shape[0]
        s, s_prime = torch.triu_indices(Nrv, Nrv, offset=1, device=eta.device)

        # Batch processing to avoid OOM
        batch_size = min(len(s), self.batch_size(Nrv))  # Adjust batch size based on memory constraints

        G_delta_0_G_delta_0_sum = torch.zeros((2 * self.Ltau, self.Ly, self.Lx),
                                              dtype=eta_ext_conj.dtype, device=eta.device)
        total_pairs = len(s)
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            s_batch = s[start_idx:end_idx]
            s_prime_batch = s_prime[start_idx:end_idx]
            
            # a = eta_ext_conj[s_batch] * eta_ext_conj[s_prime_batch]
            # b = G_eta_ext[s_batch] * G_eta_ext[s_prime_batch]
            a = torch.roll(eta_ext_conj[s_batch], shifts=a_xi, dims=-1) * torch.roll(eta_ext_conj[s_prime_batch], shifts=a_xi_prime, dims=-1)
            b = torch.roll(G_eta_ext[s_batch], shifts=b_G_xi, dims=-1) * torch.roll(G_eta_ext[s_prime_batch], shifts=b_G_xi_prime, dims=-1)

            a_F_neg_k = torch.fft.ifftn(a, (2 * self.Ltau, self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (2 * self.Ltau, self.Ly, self.Lx), norm="forward")

            G_delta_0_G_delta_0_sum += batch_result.sum(dim=0)

        G_delta_0_G_delta_0 = G_delta_0_G_delta_0_sum / total_pairs

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


        Lx = self.Lx
        Ly = self.Ly
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
        G_d = torch.fft.ifft2(G_fft_diag_adv_idx, s=(Ly, Lx), norm="backward")  # [2*Ltau, Ly, Lx]
        G_mean_fft = G_d.mean(dim=0) * Lx*Ly  # [Ly, Lx]

        if debug:
            # Verify equivalence with reference implementation
            torch.testing.assert_close(G_mean_ref[0].real, G_mean_fft.real, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)
            torch.testing.assert_close(G_mean_ref[0], G_mean_fft, rtol=1e-3, atol=1e-3, equal_nan=True, check_dtype=False)

        return G_mean_fft.unsqueeze(0)  # Return tau=0 slice: [Ly, Lx]


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
        
        Lx = self.Lx
        Ly = self.Ly
        
        if debug:
            Ltau2 = 2 * self.Ltau

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

        # ---- vectorized version ---- #
        # Prepare all indices including tau
        tau_idx = torch.arange(self.Ltau, device=G_fft.device)
        ky_idx = torch.arange(Ly, device=G_fft.device)
        kx_idx = torch.arange(Lx, device=G_fft.device)
        py_idx = torch.arange(Ly, device=G_fft.device)
        px_idx = torch.arange(Lx, device=G_fft.device)
        
        # Create meshgrid for all dimensions at once
        tau_grid, ky_grid1, kx_grid1, ky_grid2, kx_grid2, py_grid, px_grid = torch.meshgrid(
            tau_idx, ky_idx, kx_idx, ky_idx, kx_idx, py_idx, px_idx, indexing='ij'
        )  # [Ltau, Ly, Lx, Ly, Lx, Ly, Lx]

        # Vectorized computation for all tau and p = (py, px)
        # sum_{k, k'} G[:, -k'+p, :, -k-p] * G[:, k, :, k'] e^{i * p * d}
        G1 = G_fft[tau_grid, 
               (-ky_grid2 + py_grid) % Ly, 
               (-kx_grid2 + px_grid) % Lx, 
               tau_grid,
               (-ky_grid1 - py_grid) % Ly, 
               (-kx_grid1 - px_grid) % Lx]
        G2 = G_fft[tau_grid, ky_grid1, kx_grid1,
                   tau_grid, ky_grid2, kx_grid2]

        # sum over k, k' dimensions (dimensions 1,2,3,4)
        result_fft = (G1 * G2).sum(dim=(1, 2, 3, 4))  # [Ltau, Ly, Lx]


        # # ---- semi vectorized version ---- #
        # # Vectorized computation over all (py, px) at once
        # # G_fft_tau: [Ly, Lx, Ly, Lx] for fixed tau
        # for tau in tqdm(range(self.Ltau)):
        #     # G_fft_tau: [Ly, Lx, Ly, Lx] for fixed tau
        #     G_fft_tau = G_fft[tau, :, :, tau, :, :]  # [Ly, Lx, Ly, Lx]

        #     # Prepare k, k' indices
        #     ky_idx = torch.arange(Ly, device=G_fft.device)
        #     kx_idx = torch.arange(Lx, device=G_fft.device)
        #     py_idx = torch.arange(Ly, device=G_fft.device)
        #     px_idx = torch.arange(Lx, device=G_fft.device)
            
        #     ky_grid1, kx_grid1, ky_grid2, kx_grid2, py_grid, px_grid = torch.meshgrid(
        #         ky_idx, kx_idx, ky_idx, kx_idx, py_idx, px_idx, indexing='ij'
        #     )  # [Ly, Lx, Ly, Lx, Ly, Lx]

        #     # Vectorized computation for all p = (py, px)
        #     #sum_{k, k'} G[:, -k'+p, :, -k-p] * G[:, k, :, k'] e^{i * p * d}
        #     G1 = G_fft_tau[(-ky_grid2 + py_grid) % Ly, 
        #                    (-kx_grid2 + px_grid) % Lx, 
        #                    (-ky_grid1 - py_grid) % Ly, 
        #                    (-kx_grid1 - px_grid) % Lx]
        #     G2 = G_fft_tau[ky_grid1, kx_grid1, ky_grid2, kx_grid2]

        #     # sum over k, k' dimensions (first 4 dimensions)
        #     val = (G1 * G2).sum(dim=(0, 1, 2, 3))  # [Ly, Lx]
        #     result_fft[tau] = val


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

        return G_mean_fft.unsqueeze(0)  # tau=0 slice: [Ly, Lx]


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
        = - grup (D, 0) * grup(0, D) + grup(0, 0) * delta_{D, 0}, where Delta = j - i
        = - GD0_G0D + G_D0 * delta_{D, 0}

        spsm: [Ly, Lx]
        """
        spsm = -GD0_G0D  # [Ltau, Ly, Lx]
        spsm[0, 0, 0] += GD0[0, 0, 0]  
        return spsm.real[0]  # Return only tau=0 slice: [Ly, Lx]
    
        # spsm = -GD0_G0D[0]  # [Ly, Lx]
        # spsm[0, 0] += GD0[0, 0, 0]
        # spsm = spsm.real
        # return spsm
    
    def spsm_r_minus_bg(self, GD0_G0D, GD0):
        """
        spsm: [Ly, Lx]
        """
        spsm = self.spsm_r(GD0_G0D, GD0)
        # spsm -= (GD0[0, 0, 0]**2).abs() # This is wrong background
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

    @torch.inference_mode()
    def get_fermion_obsr(self, bosons, eta):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields
        eta: [Nrv, Ltau * Ly * Lx]

        Returns:
            spsm_r: [bs, Ly, Lx] tensor, spsm[i, j, tau] = <c^+_i c_j> * <c_i c^+_j>
            spsm_k: [bs, Ly, Lx] tensor.
        """
        bs = bosons.shape[0]
        obsrs = []
        # self.indices = indices
        # self.indices_r2 = indices_r2
        for b in range(bs):
            obsr = {}

            boson = bosons[b].unsqueeze(0)  # [1, 2, Ltau, Ly, Lx]
            self.set_eta_G_eta(boson, eta)

            # obsr.update(self.get_spsm_per_b())
            obsr.update(self.get_dimer_dimer_per_b2())

            obsrs.append(obsr)

        # Consolidate the obsrs according to the key of the obsrs. For each key, the tensor is of shape [Ly, Lx]. Stack them to get [bs, Ly, Lx].
        keys = obsrs[0].keys()
        consolidated_obsr = {}
        for key in keys:
            consolidated_obsr[key] = torch.stack([obsr[key] for obsr in obsrs], dim=0)

        return consolidated_obsr

    @torch.inference_mode()
    def get_fermion_obsr_compile(self, bosons, eta):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields
        eta: [Nrv, Ltau * Ly * Lx]

        Returns:
            spsm_r: [bs, Ly, Lx] tensor, spsm[i, j, tau] = <c^+_i c_j> * <c_i c^+_j>
            spsm_k: [bs, Ly, Lx] tensor.
        """
        bs = bosons.shape[0]
        obsrs = []
        for b in range(bs):
            obsr = {}

            boson = bosons[b].unsqueeze(0)  # [1, 2, Lx, Ly, Ltau]
            self.set_eta_G_eta(boson, eta)

            # obsr.update(self.get_spsm_per_b2())
            obsr.update(self.get_bond_bond_per_b2(boson))
            # obsr.update(self.get_dimer_dimer_per_b2())

            obsrs.append(obsr)

        # Consolidate the obsrs according to the key of the obsrs. For each key, the tensor is of shape [Ly, Lx]. Stack them to get [bs, Ly, Lx].
        keys = obsrs[0].keys()
        consolidated_obsr = {}
        for key in keys:
            consolidated_obsr[key] = torch.stack([obsr[key] for obsr in obsrs], dim=0)
        
        return consolidated_obsr
    
    @torch.inference_mode()
    def get_fermion_obsr_gt(self, bosons):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields
        eta: [Nrv, Ltau * Ly * Lx]

        Returns:
            spsm_r: [bs, Ly, Lx] tensor, spsm[i, j, tau] = <c^+_i c_j> * <c_i c^+_j>
            spsm_k: [bs, Ly, Lx] tensor.
        """
        bs = bosons.shape[0]
        obsrs = []
        for b in range(bs):
            obsr = {}

            boson = bosons[b].unsqueeze(0)  # [1, 2, Ltau, Ly, Lx]
            # self.set_eta_G_eta(boson, eta)

            # obsr.update(self.get_spsm_per_b())
            obsr.update(self.get_dimer_dimer_per_b_gt(boson))

            obsrs.append(obsr)
        
        # Consolidate the obsrs according to the key of the obsrs. For each key, the tensor is of shape [Ly, Lx]. Stack them to get [bs, Ly, Lx].
        keys = obsrs[0].keys()
        consolidated_obsr = {}
        for key in keys:
            consolidated_obsr[key] = torch.stack([obsr[key] for obsr in obsrs], dim=0)
        
        return consolidated_obsr

    def get_spsm_per_b(self):
        GD0_G0D = self.G_delta_0_G_0_delta_ext_batch() # [Ltau, Ly, Lx]
        GD0 = self.G_delta_0_ext() # [Ltau, Ly, Lx]

        spsm_r = self.spsm_r(GD0_G0D, GD0)  # [Ly, Lx]
        spsm_k_abs = self.spsm_k(spsm_r).abs()  # [Ly, Lx]

        # Output
        obsr = {}
        obsr['spsm_r'] = spsm_r
        obsr['spsm_k_abs'] = spsm_k_abs
        return obsr

    def get_spsm_per_b2(self):
        if self.cuda_graph_se:
            spsm_r = self.spsm_graph_runner(self.eta, self.G_eta)
        else:
            spsm_r = self.spsm_r_util(self.eta, self.G_eta)

        # torch.testing.assert_close(spsm_r, spsm_r_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)

        spsm_k = torch.fft.ifft2(spsm_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        spsm_k = self.reorder_fft_grid2(spsm_k)  # [Ly, Lx]

        # Output
        obsr = {}
        obsr['spsm_r'] = spsm_r.real
        obsr['spsm_k'] = spsm_k.real
        return obsr

    def spsm_r_util(self, eta, G_eta):
        # eta: [Nrv, Ltau * Ly * Lx]
        # G_eta: [Nrv, Ltau, Ly, Lx]
        GD0_G0D = self.GD0_G0D_func(eta, G_eta) # [Ltau, Ly, Lx]
        GD0 = self.GD0_func(eta, G_eta) # [Ltau, Ly, Lx]
        spsm = -GD0_G0D  # [Ltau, Ly, Lx]
        spsm[0, 0, 0] += GD0[0, 0, 0]
        spsm_r = spsm[0]  # Return only tau=0 slice: [Ly, Lx]
        return spsm_r

    def get_bond_bond_per_b2(self, boson):
        # eta: [Nrv, Ltau * Ly * Lx]
        # G_eta: [Nrv, Ltau * Ly * Lx]
        # boson: [1, 2, Lx, Ly, Ltau]

        if self.cuda_graph_se:
            T1 = self.T1_graph_runner(self.eta, self.G_eta, boson)
        else:
            T1 = self.T1(self.eta, self.G_eta, boson)
        # torch.testing.assert_close(T1, T1_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)

        if self.cuda_graph_se:
            T21 = self.T21_graph_runner(self.eta, self.G_eta, boson)
        else:
            T21 = self.T21(self.eta, self.G_eta, boson)
        # torch.testing.assert_close(T2, T2_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)

        if self.cuda_graph_se:
            T2 = self.T2_graph_runner(self.eta, self.G_eta, boson)
        else:
            T2 = self.T2(self.eta, self.G_eta, boson)
        # torch.testing.assert_close(T3, T3_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)

        if self.cuda_graph_se:
            T4 = self.T4_graph_runner(self.eta, self.G_eta, boson)
        else:
            T4 = self.T4(self.eta, self.G_eta, boson)
        # torch.testing.assert_close(T4, T4_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)

        if self.cuda_graph_se:
            Tvv, Tv = self.Tvv_graph_runner(self.eta, self.G_eta, boson)
        else:
            Tvv_ref, Tv_ref = self.Tvv(self.eta, self.G_eta, boson)

        torch.testing.assert_close(Tvv, Tvv_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)
        torch.testing.assert_close(Tv, Tv_ref, rtol=1e-5, atol=1e-5, equal_nan=True, check_dtype=False)


        BB_r = (T1 + T21 + T2 + T4) * 2 + Tvv * 4
        BB_k = torch.fft.ifft2(BB_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        BB_k = self.reorder_fft_grid2(BB_k)  # [Ly, Lx]

        # Output
        obsr = {}
        obsr['BB_r'] = BB_r.real    # intensive quantity
        obsr['B_r'] = Tv.real       # intensive quantity
        obsr['BB_k'] = BB_k.real    # need a renormalization of 1/Vs
        return obsr
    
    def bond_corr(self, BB_r_mean, B_r_mean):
        # BB_r_mean: [Ly, Lx], mean over configurations
        # B_r_mean: [Ly, Lx], mean over configurations
        vi = B_r_mean   
        v_F_neg_k = torch.fft.ifftn(vi, (self.Ly, self.Lx), norm="backward")
        v_F = torch.fft.fftn(vi, (self.Ly, self.Lx), norm="forward")
        v_bg = torch.fft.ifftn(v_F_neg_k * v_F, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        bond_corr = BB_r_mean - 4 * v_bg 
        return bond_corr # Delta: [Ly, Lx]
    
    def get_spsm(self, bosons, eta):
        bs, _, Lx, Ly, Ltau = bosons.shape
        spsm_r = torch.zeros((bs, Ly, Lx), dtype=self.dtype, device=self.device)
        spsm_k_abs = torch.zeros((bs, Ly, Lx), dtype=self.dtype, device=self.device)
        # szsz = torch.zeros((bs, Lx, Ly), dtype=self.dtype, device=self.device)
        
        for b in range(bs):
            boson = bosons[b].unsqueeze(0)  # [1, 2, Lx, Ly, Ltau]

            self.set_eta_G_eta(boson, eta, b)
            GD0_G0D = self.G_delta_0_G_0_delta_ext_batch(b) # [Ltau, Ly, Lx]
            GD0 = self.G_delta_0_ext(b) # [Ltau, Ly, Lx]

            spsm_r[b] = self.spsm_r(GD0_G0D, GD0)  # [Ly, Lx]
            # spsm_r[b] = self.spsm_r_minus_bg(GD0_G0D, GD0)  # [Ly, Lx]
            spsm_k_abs[b] = self.spsm_k(spsm_r[b]).abs()  # [Ly, Lx]

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
            spsm_r[b] = spsm_r_per_b
            # spsm_r_mgb = self.spsm_r_minus_bg(GD0_G0D, GD0)  # [Ly, Lx]
            spsm_k_abs[b] = self.spsm_k(spsm_r_per_b).abs()  # [Ly, Lx]

            # szsz[b] = 0.5 * spsm[b]

        obsr = {}
        obsr['spsm_r'] = spsm_r
        obsr['spsm_k_abs'] = spsm_k_abs
        return obsr

    def get_dimer_dimer_per_b(self):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields

        Returns:
            DD_r: [Ly, Lx] tensor
            DD_k: [Ly, Lx] tensor
        """
        z2 = self.hmc_sampler.Nf**2 - 1
        z4 = z2*z2
        z3 = self.hmc_sampler.Nf ** 3 - 2 * self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf
        z1 = -self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf
        
        # Compute green's functions
        if self.GD0_G0D is None:
            self.GD0_G0D = self.G_delta_0_G_0_delta_ext_batch() # [Ltau, Ly, Lx]
        if self.GD0 is None:
            self.GD0 = self.G_delta_0_ext() # [Ltau, Ly, Lx]

        GD0_G0D = self.GD0_G0D  # [Ltau, Ly, Lx]
        GD0 = self.GD0  # [Ltau, Ly, Lx]

        L0_lft = -self.G_delta_delta_G_0_0_ext_batch(a_xi=-1, b_G_xi=-1)
        L0_rgt = -self.G_delta_delta_G_0_0_ext_batch(a_G_xi=-1, b_xi=-1)
        L0 = z4 * (L0_lft * L0_rgt)  # [Ltau, Ly, Lx]
        print("L0:", L0.real[0])

        L1_lft = -GD0_G0D
        L1_lft[0, 0, 0] += GD0[0, 0, 0]
        L1 = z2 * (L1_lft**2) # [Ltau, Ly, Lx]
        # print("L1:", L1.real)

        L2_lft = -torch.roll(GD0_G0D, shifts=-1, dims=2)  # translate by (0, 0, -1) in (Ltau, Ly, Lx)
        L2_lft[0, 0, -1] += GD0[0, 0, 0]
        L2_rgt = -torch.roll(GD0_G0D, shifts=1, dims=2)  # translate by (0, 0, 1) in (Ltau, Ly, Lx)
        L2_rgt[0, 0, 1] += GD0[0, 0, 0]
        L2 = z2 * (L2_lft * L2_rgt) # [Ltau, Ly, Lx]
        # print("L2:", L2.real)

        L3_lft = self.G_delta_delta_G_0_0_ext_batch(a_xi=-1, b_xi=-1)  
        L3_rgt = -self.G_delta_0_G_0_delta_ext_batch(a_G_xi=-1, b_G_xi=-1)  
        L3_rgt[0, 0, -1] += GD0[0, 0, -2]
        L3 = z3 * (L3_lft * L3_rgt)  # [Ltau, Ly, Lx]
        # print("L3:", L3.real)

        L4_lft = -self.G_delta_delta_G_0_0_ext_batch(a_G_xi=-1, b_G_xi=-1)
        L4_rgt = self.G_delta_0_G_0_delta_ext_batch(a_xi=-1, b_xi=-1)
        L4 = z3 * (L4_lft * L4_rgt)  # [Ltau, Ly, Lx]
        # print("L4:", L4.real)

        L5_lft = -self.G_delta_0_G_0_delta_ext_batch(a_G_xi=-1, b_xi=-1)
        L5_lft[0, 0, 0] += GD0[0, 0, 0]
        L5_rgt = -L0_lft
        L5 = z3 * (L5_lft * L5_rgt)  # [Ltau, Ly, Lx]
        # print("L5:", L5.real)

        L6_lft = L0_rgt
        L6_rgt = self.G_delta_0_G_0_delta_ext_batch(a_xi=-1, b_G_xi=-1)
        L6 = z3 * (L6_lft * L6_rgt)  # [Ltau, Ly, Lx]
        # print("L6:", L6.real)

        L7_lft = -self.G_delta_0_G_delta_0_ext_batch(a_xi_prime=-1, b_G_xi=-1)
        L7_lft[0, 0, -1] += GD0[0, 0, 2]
        L7_rgt = self.G_0_delta_G_0_delta_ext_batch(a_G_xi_prime=-1, b_xi_prime=-1)
        L7 = z1 * (L7_lft * L7_rgt)  # [Ltau, Ly, Lx]
        # print("L7:", L7.real)

        L8_lft = self.G_0_delta_G_0_delta_ext_batch(a_G_xi_prime=-1, b_xi=-1)
        L8_rgt = -self.G_delta_0_G_delta_0_ext_batch(a_xi_prime=-1, b_G_xi_prime=-1)
        L8_rgt[0, 0, 0] += GD0[0, 0, 0]
        L8 = z1 * (L8_lft * L8_rgt)  # [Ltau, Ly, Lx]
        # print("L8:", L8.real)

        DD_r = (L0 + L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8).real[0]  # [Ly, Lx]

        # Output
        obsr = {}
        obsr['DD_r'] = DD_r

        DD_k = torch.fft.ifft2(DD_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        DD_k = self.reorder_fft_grid2(DD_k)  # [Ly, Lx]
        obsr['DD_k'] = DD_k
        return obsr
    
    def get_dimer_dimer_per_b2(self):
        """
        bosons: [bs, 2, Lx, Ly, Ltau] tensor of boson fields

        Returns:
            DD_r: [Ly, Lx] tensor
            DD_k: [Ly, Lx] tensor
        """
        z2 = self.hmc_sampler.Nf**2 - 1
        z4 = z2*z2
        z3 = self.hmc_sampler.Nf ** 3 - 2 * self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf
        z1 = -self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf

        DD_r = (
            z4 * self.L0()      # rtol=0.2 norm ok, DD_k bug. DD_r_se all 0.0076
            + z2 * self.L1()    # rtol=1.1, norm bug, DD_k bug
            + z2 * self.L2()    # rtol=1.3, norm diff, DD_k not match.  DD_r and DD_k change sign in se but not in gt
            + z3 * self.L3()    # rtol=0.8, norm ok, DD_k margin. match
            + z3 * self.L4()    # rtol=0.8, norm ok, DD_k margin. match
            + z3 * self.L5()   # rtol=2, norm diff, DD_k bug 
            + z3 * self.L6()   # rtol=0.1 norm ok, DD_k match
            + z1 * self.L7() # rtol=0.1 norm ok, DD_k match
            + z1 * self.L8() # rtol=0.05 norm ok, DD_k match
        )  # [Ly, Lx]

        # Output
        obsr = {}
        obsr['DD_r'] = DD_r.real

        DD_k = torch.fft.ifft2(DD_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        DD_k = self.reorder_fft_grid2(DD_k)  # [Ly, Lx]
        obsr['DD_k'] = DD_k.real
        return obsr

    def get_dimer_dimer_per_b_groundtruth(self, boson):
        z2 = self.hmc_sampler.Nf**2 - 1
        z4 = z2*z2
        z3 = self.hmc_sampler.Nf ** 3 - 2 * self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf
        z1 = -self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf
    
        Gij_gt = self.G_groundtruth(boson) # [Ltau * Ly * Lx, Ltau * Ly * Lx]
        GD0 = self.G_delta_0_groundtruth_ext_fft(Gij_gt)
        # Compute GcD0 from GD0 according to:
        # grupc(j, i) = -grup(i, j) for i != j, grupc(i, i) = -grup(i, i) + 1
        # Vectorized computation of GcD0 from GD0:
        # GcD0[tau, y, x] = -GD0[-tau % Ltau, -y % Ly, -x % Lx] for (tau, y, x) != (0, 0, 0)
        # and GcD0[0, 0, 0] = -GD0[0, 0, 0] + 1
        Ltau, Ly, Lx = GD0.shape
        idx_tau = torch.arange(Ltau, device=GD0.device)
        idx_y = torch.arange(Ly, device=GD0.device)
        idx_x = torch.arange(Lx, device=GD0.device)
        tau_grid, y_grid, x_grid = torch.meshgrid(idx_tau, idx_y, idx_x, indexing='ij')
        neg_tau = (-tau_grid) % Ltau
        neg_y = (-y_grid) % Ly
        neg_x = (-x_grid) % Lx
        GcD0 = -GD0[neg_tau, neg_y, neg_x]

        # Assert that GD0[neg_tau, neg_y, neg_x] == GD0[tau, y, x]
        torch.testing.assert_close(GD0[neg_tau, neg_y, neg_x], GD0, rtol=5e-2, atol=5e-2)
        GcD0[0, 0, 0] = -GD0[0, 0, 0] + 1
        GcD0 = GcD0.squeeze(0)  # [Ly, Lx]
        GD0 = GD0.squeeze(0)  # [Ly, Lx]
        torch.testing.assert_close(GcD0[neg_y[0], neg_x[0]], GcD0, rtol=3e-2, atol=5e-2)

        # Compute DD per line; below GD0 is treated as G0D
        G0D = GD0[neg_y[0], neg_x[0]]
        Gc0D = GcD0[neg_y[0], neg_x[0]]

        # grup(i,iax)*grupc(j,jax)* grupc(i,iax)  *grup(j,jax)
        L0_lft = G0D[0, 1] * Gc0D[0, 1]
        L0_rgt = Gc0D[0, 1] * G0D[0, 1]
        L0 = z4 * (L0_lft * L0_rgt)
        print("L0:", L0.real)

        # grupc(i,j)  *grup(i,j)  *grupc(iax,jax)*grup(iax,jax)*z2 
        L1_lft = Gc0D * G0D
        L1 = z2 * (L1_lft**2)
        print("L1:", L1.real.shape)

        # grupc(i,jax)*grup(i,jax)*grupc(iax,j)  *grup(iax,j)  *z2
        L2_lft = torch.roll(Gc0D, shifts=-1, dims=-1) * torch.roll(G0D, shifts=-1, dims=-1)
        L2_rgt = torch.roll(Gc0D, shifts=1, dims=-1) * torch.roll(G0D, shifts=1, dims=-1)
        L2 = z2 * (L2_lft * L2_rgt) # [Ltau, Ly, Lx]
        print("L2:", L2.real.shape)

        # grupc(i,jax)*grup(i,iax)*grup(iax,j)   *grup(j,jax)  *z3  
        L3_lft = torch.roll(Gc0D, shifts=-1, dims=-1) * G0D[0, 1]
        L3_rgt = torch.roll(G0D, shifts=1, dims=-1) * G0D[0, 1]
        L3 = z3 * (L3_lft * L3_rgt)  # [Ltau, Ly, Lx]
        print("L3:", L3.real.shape)

        # grupc(i,iax)*grup(i,jax)*grup(j,iax)   *grup(jax,j)  *z3
        L4_lft = Gc0D[0, 1] * torch.roll(G0D, shifts=-1, dims=-1)
        L4_rgt = torch.roll(GD0, shifts=1, dims=-1) * GD0[0, 1]
        L4 = z3 * (L4_lft * L4_rgt)
        print("L4:", L4.real.shape)

        # grupc(i,j)  *grup(i,iax)*grup(iax,jax) *grup(jax,j)  *z3
        L5_lft = Gc0D * G0D[0, 1]
        L5_rgt = G0D * GD0[0, 1]
        L5 = z3 * (L5_lft * L5_rgt)
        print("L5:", L5.real.shape)

        # grupc(i,iax)*grup(i,j)  *grup(jax,iax) *grup(j,jax)  *z3
        L6_lft = Gc0D[0, 1] * G0D
        L6_rgt = GD0 * G0D[0, 1]
        L6 = z3 * (L6_lft * L6_rgt)
        print("L6:", L6.real.shape)

        # grupc(i,jax)*grup(i,j)  *grup(iax,jax) *grup(j,iax)  *z1
        L7_lft = torch.roll(Gc0D, shifts=-1, dims=-1) * G0D
        L7_rgt = G0D * torch.roll(GD0, shifts=1, dims=-1)
        L7 = z1 * (L7_lft * L7_rgt)
        print("L7:", L7.real.shape)

        # grupc(i,j)  *grup(i,jax)*grup(jax,iax) *grup(iax,j)  *z1
        L8_lft = Gc0D * torch.roll(G0D, shifts=-1, dims=-1)
        L8_rgt = GD0 * torch.roll(G0D, shifts=1, dims=-1)
        L8 = z1 * (L8_lft * L8_rgt)
        print("L8:", L8.real.shape)

        DD_r = (L0 + L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8).real  # [Ly, Lx]

        # Output
        obsr = {}
        obsr['DD_r'] = DD_r

        DD_k = torch.fft.ifft2(DD_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        DD_k = self.reorder_fft_grid2(DD_k)  # [Ly, Lx]
        obsr['DD_k'] = DD_k
        return obsr

    def reset_cache(self):
        """
        Reset the cache for Green's functions.
        """
        self.GD0_G0D = None
        self.GD0 = None
    
    def get_dimer_dimer_per_b_gt(self, boson):
        """
        bosons: [2, Lx, Ly, Ltau] tensor of boson fields

        """
        Ltau, Ly, Lx = self.Ltau, self.Ly, self.Lx
        N = Ly * Lx

        Gij_gt = self.G_groundtruth(boson) # [Ltau*Ly*Lx, Ltau*Ly*Lx]
        # Take the equal-tau subtensor as G_et using indexing
        Gij_gt_reshaped = Gij_gt.view(Ltau, Ly*Lx, Ltau, Ly*Lx)
        G_eqt = Gij_gt_reshaped[torch.arange(Ltau), :, torch.arange(Ltau), :]
        assert G_eqt.shape == (Ltau, N, N), f"G_eqt shape mismatch: {G_eqt.shape}"
        
        # G_eqt_se = self.G_eqt_se()  # [Ltau, vs, vs]
        # G_eqt = G_eqt_se

        Gc_eqt = torch.zeros_like(G_eqt)
        for i in range(N):
            for j in range(N):
                Gc_eqt[:, i, j] = -G_eqt[:, j, i]
            Gc_eqt[:, i, i] += 1

        Gc, G = Gc_eqt, G_eqt  # [Ltau, vs, vs]
        
        # # Check that G_eqt_se is close to G
        # G_eqt_se = self.G_eqt_se().mean(dim=0)  # [vs, vs]
        # G_eqt = G_eqt.mean(dim=0)

        # try:
        #     torch.testing.assert_close(G_eqt_se, G_eqt, rtol=1e-2, atol=1e-2)
        # except Exception as e:
        #     print("G_eqt_se and G are not close:", e)

        # print("G_eqt_se shape:", G_eqt_se.shape)
        # print("G shape:", G_eqt.shape)

        # # Print out some values for inspection
        # print("G_eqt_se.real[0, :10]:\n", G_eqt_se.real[0, :10])
        # print("G_eqt.real[0, :10]:\n", G_eqt.real[0, :10])

        # diff = G_eqt_se - G_eqt
        # atol = diff.abs().max()
        # print("Max absolute difference between G_eqt_se and G (atol):", atol.item())

        # # Compute max relative error (rtol)
        # denom = G_eqt.abs().clamp_min(1e-12)
        # rtol = (diff.abs() / denom).max()
        # print("Max relative difference between G_eqt_se and G (rtol):", rtol.item())

        # dbstop = 1

        z2 = self.hmc_sampler.Nf**2 - 1
        z4 = z2*z2
        z3 = self.hmc_sampler.Nf ** 3 - 2 * self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf
        z1 = -self.hmc_sampler.Nf + 1/self.hmc_sampler.Nf

        # Compute the four-point green's function using the ground truth
        # G_delta_0_G_delta_0 = mean_i G_{i+d, i} G_{i+d, i}
        DD_r = torch.empty((N, N), dtype=G_eqt.dtype, device=G_eqt.device)
        for i in range(N):
            for d in range(N):
                y, x = unravel_index(torch.tensor(i, dtype=torch.int64, device=self.device), (Ly, Lx))
                dy, dx = unravel_index(torch.tensor(d, dtype=torch.int64, device=self.device), (Ly, Lx))

                j = ravel_multi_index(
                    ((y + dy) % Ly, (x + dx) % Lx),
                    (Ly, Lx)
                )

                # Compute iax and jax indices (x+1 with periodic boundary)
                iax = ravel_multi_index((y, (x + 1) % Lx), (Ly, Lx))
                jax = ravel_multi_index(((y + dy) % Ly, ((x + dx) + 1) % Lx), (Ly, Lx))

                DD_r[i, d] = (
                    Gc[:, i, iax] * G[:, i, iax] * Gc[:, j, jax] * G[:, j, jax] * z4
                    + Gc[:, i, j] * G[:, i, j] * Gc[:, iax, jax] * G[:, iax, jax] * z2
                    + Gc[:, i, jax] * G[:, i, jax] * Gc[:, iax, j] * G[:, iax, j] * z2
                    + Gc[:, i, jax] * G[:, i, iax] * G[:, iax, j] * G[:, j, jax] * z3
                    + Gc[:, i, iax] * G[:, i, jax] * G[:, j, iax] * G[:, jax, j] * z3
                    + Gc[:, i, j] * G[:, i, iax] * G[:, iax, jax] * G[:, jax, j] * z3
                    + Gc[:, i, iax] * G[:, i, j] * G[:, jax, iax] * G[:, j, jax] * z3
                    + Gc[:, i, jax] * G[:, i, j] * G[:, iax, jax] * G[:, j, iax] * z1
                    + Gc[:, i, j] * G[:, i, jax] * G[:, jax, iax] * G[:, iax, j] * z1
                ).mean(dim=0) # mean over tau

        # Output
        obsr = {}
        DD_r = DD_r.mean(dim=0).view(Ly, Lx) # [Ly, Lx], mean over i
        obsr['DD_r'] = DD_r.real

        DD_k = torch.fft.ifft2(DD_r, (self.Ly, self.Lx), norm="forward")  # [Ly, Lx]
        DD_k = self.reorder_fft_grid2(DD_k)  # [Ly, Lx]
        obsr['DD_k'] = DD_k.real
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

