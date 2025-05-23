import torch
import os
import sys  

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
        self.Nrv = 20
        self.green_four = None
        self.green_two = None

        self.Lx = hmc_sampler.Lx
        self.Ly = hmc_sampler.Ly    
        self.Ltau = hmc_sampler.Ltau
        self.dtau = hmc_sampler.dtau

        self.Vs = hmc_sampler.Vs

        self.use_cuda_graph = hmc_sampler

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
    
        dbstop = 1


    def four_point_green(self, boson):
        """
        Compute the four-point green's function

        boson: [bs=1, 2, Lx, Ly, Ltau]
        """
        # Compute the four-point green's function
        # G_ij ~ (G eta)_i eta_j
        # G_ij G_kl ~ (G eta)_i eta_j (G eta')_k eta'_l
        boson = boson.permute([0, 4, 3, 2, 1]).reshape(self.bs, -1).repeat(self.Nrv, 1)  # [Nrv, Lx * Ly * Ltau]

        eta = self.random_vec_bin()  # [Nrv, Lx * Ly * Ltau]
        
        psudo_fermion = _C.mh_vec(boson, eta, self.Lx, self.dtau, *BLOCK_SIZE)  # [Nrv, Ltau * Ly * Lx]

        G_eta = self.hmc_sampler.Ot_inv_psi_fast(psudo_fermion, boson, None)  # [Nrv, Ltau * Ly * Lx]

        # Build augmented eta and G_eta
        # eta_ext = [eta, -eta], G_eta_ext = [G_eta, -G_eta]
        eta_ext = torch.cat([eta, -eta], dim=1)
        G_eta_ext = torch.cat([G_eta, -G_eta], dim=1)

        # Compute the four-point green's function
        # G_delta_0_G_delta_0
        a = eta_ext[s] * eta_ext[s_prime]
        b = G_eta_ext[s] * G_eta_ext[s_prime]

        a_F = torch.fft.fft(a, dim=1, norm="orthogonal")
        # To get a_F(-k), use torch.flip on the frequency dimension
        a_F_neg_k = torch.flip(a_F, dims=[1])
        ks = torch.fft.fftfreq(a_F.shape[1])  # [Ltau * Ly * Lx]
        dbstop = 1

        b_F = torch.fft.fft(b, dim=1, norm="orthogonal")
        G_delta_0_G_delta_0 = torch.fft.ifft(a_F_neg_k * b_F, dim=1, norm="orthogonal").view(self.Nrv)




if __name__ == "__main__":  
    hmc = HmcSampler()
    se = StochaticEstimator(hmc)
    
    se.test_orthogonality(se.random_vec_bin())

    se.test_orthogonality(se.random_vec_norm())

    



    

