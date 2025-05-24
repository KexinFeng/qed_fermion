import torch
import os
import sys  

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')
from qed_fermion.hmc_sampler_batch import HmcSampler

if torch.cuda.is_available():
    from qed_fermion import _C 
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = (4, 8)
print(f"BLOCK_SIZE: {BLOCK_SIZE}")

class DetSign:
    # https://chatgpt.com/share/683166af-7dc0-8011-b7da-eec66d263eb2
    
    def __init__(self, hmc_sampler):
        self.hmc = hmc_sampler
        self.green_four = None
        self.green_two = None

        self.Lx = hmc_sampler.Lx
        self.Ly = hmc_sampler.Ly    
        self.Ltau = hmc_sampler.Ltau
        self.dtau = hmc_sampler.dtau

        self.Vs = hmc_sampler.Vs

        self.use_cuda_graph = hmc_sampler

    def layer_monopole(self):
        boson = self.hmc.boson  # [bs, 2, Lx, Ly, Ltau]
        boson[..., 0] = 0


if __name__ == "__main__":  
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    hmc = HmcSampler()
    hmc.Lx = 2
    hmc.Ly = 2
    hmc.Ltau = 4

    hmc.bs = 1
    hmc.reset()