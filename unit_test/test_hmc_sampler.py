import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.hmc_sampler import HmcSampler


def test_greens_function():
    hmc = HmcSampler()
    a, x, y, tau = 2, 4, 4, 10  # Example dimensions

    hmc.Lx, hmc.Ly, hmc.Ltau = 4, 4, 10
    hmc.num_tau = 5

    boson = torch.randn(a, x, y, tau)  # Random tensor
    result = hmc.greens_function(boson)
    print(result.shape)


if __name__ == '__main__':
    test_greens_function()