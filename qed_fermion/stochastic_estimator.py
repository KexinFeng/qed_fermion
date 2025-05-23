import torch
import os
import sys  

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')
from qed_fermion.hmc_sampler_batch import HmcSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def random_vec_bin(self):  
        """
        Generate Nrv iid random variables
    
        return: Nrv random complex vectors with entries from {1, -1, 1j, -1j} / sqrt(2), [Nrv, Lx * Ly * Ltau]
        """
        # Generate Nrv random complex vectors with entries from {1, -1, 1j, -1j} / sqrt(2)
        Lx, Ly, Ltau = self.hmc_sampler.Lx, self.hmc_sampler.Ly, self.hmc_sampler.Ltau

        N_site = Lx * Ly * Ltau

        real_part = torch.randint(0, 2, (self.Nrv, N_site), dtype=torch.float32, device=device) * 2 - 1  # { -1, 1 }
        imag_part = torch.randint(0, 2, (self.Nrv, N_site), dtype=torch.float32, device=device) * 2 - 1  # { -1, 1 }

        rand_vec = (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(2.0, device=device))

        return rand_vec

    def random_vec_norm(self):
        """
        Generate Nrv iid real Gaussian random variables

        return: Nrv random real vectors with entries ~ N(0, 1/sqrt(2)), [Nrv, Lx * Ly * Ltau]
        """
        Lx, Ly, Ltau = self.hmc_sampler.Lx, self.hmc_sampler.Ly, self.hmc_sampler.Ltau
        N_site = Lx * Ly * Ltau

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

        

if __name__ == "__main__":  
    hmc = HmcSampler()
    se = StochaticEstimator(hmc)
    
    se.test_orthogonality(se.random_vec_bin())

    se.test_orthogonality(se.random_vec_norm())

    



    

