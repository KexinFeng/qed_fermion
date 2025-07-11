import matplotlib.pyplot as plt

plt.ion()

import numpy as np
import math

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator
from qed_fermion.stochastic_estimator import StochaticEstimator

from load_write2file_convert import time_execution

# Add partition parameters

# Lx = int(os.getenv("Lx", '6'))
# print(f"Lx: {Lx}")
# Ltau = int(os.getenv("Ltau", '60'))
# print(f"Ltau: {Ltau}")

# asym = Ltau / Lx * 0.1

# part_size = 500
# start_dqmc = 5000
# end_dqmc = 10000

start_dist = 1
step_dist = 2 
y_diplacement = lambda x: 0

# Only use r > 0 for log-log fit to avoid log(0)
lw = 1 if start_dist == 0 else 0
up = 10

suffix = None
if start_dist == 1 and step_dist == 2:
    suffix = "odd"
elif start_dist == 0 and step_dist == 2:
    suffix = "even"
else:
    if y_diplacement(1) == 0:
        suffix = "all"
    else:
        suffix = "diag"

@time_execution
def plot_spsm(Lsize=(6, 6, 10), bs=5, ipair=(0, 1)):
    Lx, Ly, Ltau = Lsize

    color_idx, color_idx2 = ipair

    Js = [1, 1.5, 2, 2.3, 2.5, 3]
    Js = [1]
    r_afm_values = []
    r_afm_errors = []
    spin_order_values = []
    spin_order_errors = []
    spin_r_values = []
    spin_r_errors = []

    vs = Lx**2
    
    # plt.figure(figsize=(8, 6))
    
    for J in Js:
        # Initialize data collection arrays for this J value
        all_data = []
        
        # Calculate the number of parts
        num_parts = math.ceil((end_dqmc - start_dqmc) / part_size)
        
        # Loop through all batches and parts
        for bid in range(bs):
            # if not bid == 0: continue
            for part_id in range(num_parts):
                input_folder = root_folder + f"/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}_bid{bid}_part_{part_id}_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"
                name = f"szsz.bin"
                ftdqmc_filename = os.path.join(input_folder, name)
                
                try:
                    part_data = np.genfromtxt(ftdqmc_filename)
                    all_data.append(part_data)
                    print(f'Loaded ftdqmc data: {ftdqmc_filename}')
                except (FileNotFoundError, ValueError) as e:
                    raise RuntimeError(f'Error loading {ftdqmc_filename}: {str(e)}') from e
        
        # Combine all parts' data
        data = np.concatenate(all_data)
        data = data.reshape(bs, -1, vs, 4)
        # data has shape [num_sample, vs, 4], where the last dim has entries: kx, ky, val, error. 
        # [num_sample]
        szsz_k = torch.tensor(data[:, :, :, 2]).view(bs, -1, Ly, Lx)
        szsz_k = StochaticEstimator.reorder_fft_grid2_inverse(szsz_k)
        szsz_r = torch.fft.fft2(szsz_k, dim=(-2, -1)).real
        r_afm = 1 - data[..., 1, 2] / data[..., 0, 2]

        szsz_r = szsz_r.view(bs, -1, Ly, Lx).numpy()

        # spin order
        spin_order = np.mean(data[..., 0, 2], axis=1)
        spin_order_err = np.mean(np.abs(data[..., 0, 3]), axis=1)
        spin_order_values.append(spin_order)
        spin_order_errors.append(spin_order_err)
        
        spin_r = np.mean(szsz_r, axis=(0, 1))
        spin_r_err = np.std(szsz_r, axis=(0, 1)) / np.sqrt(np.prod(szsz_r.shape[:2]))  # Standard error
        spin_r_values.append(spin_r)
        spin_r_errors.append(spin_r_err)

        # r_afm = spin_order
        rtol = data[:, :, :, 3] / data[:, :, :, 2]
        r_afm_err = abs(rtol[:, :, 0] - rtol[:, :, 1]) * (1 - r_afm)
        
        # Calculate mean and error for plotting
        r_afm_mean = np.mean(r_afm, axis=1)
        r_afm_error = np.mean(r_afm_err, axis=1)
        
        r_afm_values.append(r_afm_mean)
        r_afm_errors.append(r_afm_error)

    # --------- Plot spin_r_values --------- #
    spin_r = np.abs(spin_r_values[0]) # [Ly, Lx]
    spin_r_err = spin_r_errors[0] # [Ly, Lx]
    r_values = []
    spin_corr_values = []
    spin_corr_errors = []

    # Simplified: plot spin correlation along x-direction only (y=0)
    for r in range(start_dist, Lx // 2 + 1, step_dist):
        x = r
        y = y_diplacement(x)
        
        r_values.append(r)
        spin_corr_values.append(spin_r[y, x])
        spin_corr_errors.append(spin_r_err[y, x])
    
    # Plot spin correlation vs distance for this lattice size (log-log with linear fit)
    color = f"C{color_idx}"
    # Only use r > 0 for log-log fit to avoid log(0)
    # lw = 1 if start_dist == 0 else 0
    # up = 8
    r_fit = np.array(r_values[lw: up])
    spin_corr_fit = np.array(spin_corr_values[lw: up])
    # spin_corr_err_fit = np.array(spin_corr_errors[lw:up])

    # Linear fit in log-log space
    log_r = np.log(r_fit)
    log_corr = np.log(spin_corr_fit)
    coeffs = np.polyfit(log_r, log_corr, 1)
    fit_line = np.exp(coeffs[1]) * r_fit**coeffs[0]

    # Plot data and fit in log-log space
    plt.errorbar(r_values[0:], spin_corr_values[0:], yerr=spin_corr_errors[0:], 
                    linestyle='', marker='o', lw=1.5, color=color, 
                    label=f'hmc_{Lx}x{Ltau}', markersize=8, alpha=0.8)
    plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
                label=f'hmc L={Lx}: y~x^{coeffs[0]:.2f}')


    # =========== Load dqmc and plot =========== #
    dqmc_filename = dqmc_folder + f"/tuning_js_sectune_l{Lx}_spin_order.dat"
    dqmc_filename = dqmc_folder + f"/l{Lx}b{Lx}js1.0jpi1.0mu0.0nf2_dqmc_bin.dat"
    szsz_k_dqmc = np.genfromtxt(dqmc_filename)
    szsz_k_dqmc = torch.tensor(szsz_k_dqmc[:, 2]).view(Ly+1, Lx+1)
    szsz_k_dqmc = szsz_k_dqmc[:-1, :-1] 
    szsz_k_dqmc = StochaticEstimator.reorder_fft_grid2_inverse(szsz_k_dqmc)
    szsz_r = torch.fft.fft2(szsz_k_dqmc, dim=(-2, -1)).real
    
    # --------- Plot spin_r_values --------- #
    spin_r = np.abs(szsz_r) # [Ly, Lx]
    r_values = []
    spin_corr_values = []
    spin_corr_errors = []

    # Simplified: plot spin correlation along x-direction only (y=0)
    for r in range(start_dist, Lx // 2 + 1, step_dist):
        x = r
        y = y_diplacement(x)
        
        r_values.append(r)
        spin_corr_values.append(spin_r[y, x])
    
    # Plot spin correlation vs distance for this lattice size (log-log with linear fit)
    color = f"C{color_idx + 1}"
    # Only use r > 0 for log-log fit to avoid log(0)
    # lw = 1 if start_dist == 0 else 0
    # up = 8
    r_fit = np.array(r_values[lw: up])
    spin_corr_fit = np.array(spin_corr_values[lw: up])
    # spin_corr_err_fit = np.array(spin_corr_errors[lw:up])

    # Linear fit in log-log space
    log_r = np.log(r_fit)
    log_corr = np.log(spin_corr_fit)
    coeffs = np.polyfit(log_r, log_corr, 1)
    fit_line = np.exp(coeffs[1]) * r_fit**coeffs[0]

    # Plot data and fit in log-log space
    plt.errorbar(r_values[0:], spin_corr_values[0:], yerr=None, 
                    linestyle='', marker='o', lw=1.5, color=color, 
                    label=f'dqmc_{Lx}x{Ltau}', markersize=8, alpha=0.8)
    plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
                label=f'dqmc L={Lx}: y~x^{coeffs[0]:.2f}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance r (lattice units)', fontsize=14)
    plt.ylabel('Spin-Spin Correlation $\\langle S(0) S(r) \\rangle$', fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    dbstop = 1


if __name__ == '__main__':
    batch_size = 2

    plt.figure(figsize=(8, 6))
    for idx, Lx in enumerate([8, 10]):
        Ltau = Lx * 10

        asym = Ltau / Lx * 0.1

        part_size = 500
        start_dqmc = 5000
        end_dqmc = 10000

        root_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run6_{Lx}_{Ltau}/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_kexin_hk_avg/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810/piflux_B0.0K1.0_tuneJ_b1l_kexin_hk/sq_szsz"

        plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size, ipair=(2*idx, 2*idx + 1))
        dbstop = 1

    # # Plot setting
    plt.xlabel('Distance r (lattice units)', fontsize=14)
    plt.ylabel('Spin-Spin Correlation $\\langle S(0) S(r) \\rangle$', fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Set log scales first
    plt.xscale('log')
    plt.yscale('log')
    
    plt.show(block=False)
    # Save plot
    save_dir = os.path.join(script_path, f"./figures/spin_order_r_{suffix}")
    os.makedirs(save_dir, exist_ok=True) 
    file_name = "spin_order_r"
    file_path = os.path.join(save_dir, f"{file_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")




