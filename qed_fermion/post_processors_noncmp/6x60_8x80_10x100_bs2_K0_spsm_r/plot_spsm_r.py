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
step_dist = 1
y_diplacement = lambda x: 0

# Only use r > 0 for log-log fit to avoid log(0)
lw = 0 if start_dist == 1 else 1
up = 3

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

def plot_spsm(Lsize=(6, 6, 10), bs=5, ipair=(0, 1), J=1):
    Lx, Ly, Ltau = Lsize

    color_idx, color_idx2 = ipair

    Js = [1, 1.5, 2, 2.3, 2.5, 3]
    Js = [J]

    vs = Lx**2

    # =========== Load dqmc and plot =========== #
    dqmc_filename = dqmc_folder + f"/tuning_js_sectune_l{Lx}_spin_order.dat"
    dqmc_filename = dqmc_folder + f"/l{Lx}b{Lx}js{J:.1f}jpi0.0mu0.0nf2_dqmc_bin.dat"
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
    for r in range(start_dist, Lx, step_dist):
        x = r
        y = y_diplacement(x)
        
        r_values.append(r)
        spin_corr_values.append(spin_r[y, x] * 2 / vs)
    
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
                    label=f'dqmc_{Lx}x{Ltau}_J{J}', markersize=8, alpha=0.8)
    plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
                label=f'dqmc L={Lx}: y~x^{coeffs[0]:.2f}')

    # ===== Load hmc and plot ===== #
    xs = []
    ys = []
    yerrs = []
    for J in Js:
        filename = hmc_folder + f"/ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_{J}_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_False_step_10000.pt"
        data = torch.load(filename, map_location='cpu')
        spsm_r_list = data['spsm_r_list'][start_dqmc:end_dqmc]
        ys.append(spsm_r_list.mean(axis=(0, 1)))
        yerrs.append(
            std_root_n(spsm_r_list.numpy(), axis=0, lag_sum=1)[0]
        )

        # # Test spsm_k lag_sum, which is at least 5000
        # gamma0 = std_root_n(spsm_k_list.numpy(), axis=0, lag_sum=1)[1] * spsm_k_list.shape[0]**0.5
        # total_err = init_convex_seq_estimator(data['spsm_k_list'][:, 1])
        # print(f'J={J}, lag_sum: {(total_err/gamma0)**2}')
        
        xs.append(J)

    # Collect data at r
    spin_r = np.abs(ys[0]) # [Ly, Lx]
    spin_r_err = yerrs[0] # [Ly, Lx]
    r_values = []
    spin_corr_values = []
    spin_corr_errors = []

    # Simplified: plot spin correlation along x-direction only (y=0)
    for r in range(start_dist, Lx, step_dist):
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
                    label=f'hmc_{Lx}x{Ltau}_J{J}', markersize=8, alpha=0.8)
    plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
                label=f'hmc L={Lx}: y~x^{coeffs[0]:.2f}')

    # ===== Final plot settings ===== #
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
    
    # for J in [1, 1.5, 2, 2.3, 2.5, 3]:
    for J in [1]:
        plt.figure(figsize=(8, 6))
        for idx, Lx in enumerate([8, 10]):
            Ltau = Lx * 10

            asym = Ltau / Lx * 0.1

            part_size = 500
            start_dqmc = 5000
            end_dqmc = 10000

            # hmc_folder = f"/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmp_6810/K0_deprecated/hmc_check_point_noncmp_bench_K0"
            hmc_folder = f"/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmp_6810/hmc_check_point_noncmp_bench_K0_sup/"
            # root_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run6_{Lx}_{Ltau}/"
            dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_kexin_hk_avg/"
            dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810_nc/piflux_B0.0K0.0_tuneJ_b{asym:.1g}l_noncompact_kexin_hk/sq_szsz"

            plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size, ipair=(2*idx, 2*idx + 1), J=J)
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
        save_dir = os.path.join(script_path, f"./figures/spin_order_r_{suffix}_J{J}")
        os.makedirs(save_dir, exist_ok=True) 
        file_name = "spin_order_r"
        file_path = os.path.join(save_dir, f"{file_name}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")




