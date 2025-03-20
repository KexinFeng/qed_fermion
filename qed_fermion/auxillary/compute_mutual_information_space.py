import torch
import numpy as np
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.local_sampler_batch import LocalUpdateSampler


def compute_pxy(boson, x_vals, y_vals, tau_input, get_detM, action_boson_tau_cmp=None):
    """
    Compute the normalized probability distribution p(x, y).
    """
    pxy = torch.zeros(len(x_vals), len(y_vals), device=boson.device)

    # Create a grid of x and y values
    x_grid, y_grid = torch.meshgrid(x_vals, y_vals, indexing='ij')
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Clone boson for each (x, y) pair along the batch dimension
    boson_batch = boson.repeat(len(x_flat), *([1] * (boson.ndim - 1)))
    boson_batch[:, 0, 0, 0, 0] += x_flat
    boson_batch[:, 0, tau_input, 0, 0] += y_flat

    # Compute determinant for each boson in the batch
    weight = torch.real(get_detM(boson_batch)) ** 2 
    weight *= 1 if action_boson_tau_cmp is None else torch.exp(-action_boson_tau_cmp(boson_batch))
    # weight2 = torch.exp(-action_boson_tau_cmp(boson_batch))
    # print(weight1)
    # print(weight2)

    # weight = weight1 * weight2 

    # Reshape the results back to the shape of pxy
    pxy = weight.view(len(x_vals), len(y_vals))

    pxy /= pxy.sum()  # Normalize
    return pxy


def compute_mutual_information(pxy):
    """
    Compute the mutual information I(X, Y) in base 2.
    """
    px = pxy.sum(dim=1)  # Marginalize over y
    py = pxy.sum(dim=0)  # Marginalize over x

    log_base2 = lambda x: torch.log(x + 1e-12) / torch.log(torch.tensor(2.0))

    S_X = -(px * log_base2(px)).sum()
    S_Y = -(py * log_base2(py)).sum()
    S_XY = -(pxy * log_base2(pxy)).sum()

    return S_X + S_Y - S_XY

Js = [0.01, 1, 1000, 10**5]
Js = [10**5]
dtaus = [0.01, 0.1, 0.5]
for J in Js:
    for dtau in reversed(dtaus):
        # print(f'\n------- J={J} -------')
        print(f'------- dtau={dtau} -------')
        # Initialize parameters
        lmc = LocalUpdateSampler(J=J, bs=1)
        lmc.Lx, lmc.Ly, lmc.Ltau = 10, 10, 10
        lmc.dtau = dtau
        lmc.reset()
        
        lmc.initialize_boson_staggered_pi()
        boson_pi = lmc.boson.clone()
        # lmc.initialize_boson()
        # boson_randn = lmc.boson.clone()
        num_sample = 100

        lmc.bs = num_sample * num_sample
        # Define x and y values in the range [-2π, 2π]
        x_vals = torch.linspace(-2 * np.pi, 2 * np.pi, num_sample)
        y_vals = torch.linspace(-2 * np.pi, 2 * np.pi, num_sample)

        boson_str = ['boson_pi', 'boson_randn']
        for i, boson in enumerate([boson_pi]):
            # Compute mutual information for various tau_input values
            results = {}
            for distance_input in range(1, 5):  # tau_input in range (0, 5)
                pxy = compute_pxy(boson, x_vals, y_vals, distance_input, lmc.get_detM, lmc.action_boson_tau_cmp if J < 10**4 else None)
                mutual_info = torch.real(compute_mutual_information(pxy))
                results[distance_input] = mutual_info.item()

            # Print results
            print(f"\n{boson_str[i]}")
            for distance_input, mi in results.items():
                print(f"distance_input={distance_input}, Mutual Information={mi}")


