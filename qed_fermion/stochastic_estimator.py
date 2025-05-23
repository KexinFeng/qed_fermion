import torch

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
    
    def random_vec(self):  
        """
        Generate Nrv iid random variables
        """
        # [Nrv, Lx, Ly, Ltau]
        Lx, Ly, Ltau = self.hmc_sampler.sizes
        device = self.hmc_sampler.device
        # [Nrv, 2, Lx, Ly, Ltau]
        rv = torch.randn(self.Nrv, 2, Lx//2, Ly//2, Ltau).to(device)
        # [Nrv, 2*Lx*Ly*Ltau]
        rv = rv.view(self.Nrv, -1)
        return rv

    

