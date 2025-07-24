import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np
import os
import torch
import sys

plt.ion()
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
set_default_plotting()

# Parameters for data loading and plotting
start = 6000  # Skip initial equilibration steps
sample_step = 1

# Lattice sizes to analyze
lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]

# Data folder (same as in plot_fit_spsm_r.py)
hmc_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmpK0_large1_spsm/hmc_check_point_noncmpK0_large1_spsm"

plt.figure(figsize=(8, 6))

for i, Lx in enumerate(lattice_sizes):
    Ltau = int(10 * Lx)
    Ly = Lx
    if Lx <= 40:
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_False_step_10000.pt"
    elif Lx == 46:
        Nrv = 40
        bs = 2
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs{bs}_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_{Nrv}_cmp_False_step_10000.pt"
    elif Lx == 56:
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs1_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_10000.pt"
    elif Lx == 60:
        Nstp = 6800
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_{Nstp}_bs1_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_{Nstp}.pt"
    else:
        continue

    hmc_filename = os.path.join(hmc_folder, hmc_file)
    if not os.path.exists(hmc_filename):
        print(f"File not found: {hmc_filename}")
        continue

    res = torch.load(hmc_filename, map_location='cpu')
    print(f'Loaded: {hmc_filename}')

    # G_list: [timesteps, batch_size, num_tau+1]
    G_list = res['G_list']
    hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    end = int(hmc_match.group(1))
    hmc_match_bs = re.search(r'bs(\d+)', hmc_filename)
    bs = int(hmc_match_bs.group(1)) if hmc_match_bs else 1

    seq_idx = np.arange(start, end, sample_step)
    batch_idx = np.arange(bs)

    # Compute mean and std over [timesteps, batch_size]
    G_mean = G_list[seq_idx][:, batch_idx].numpy().mean(axis=(0, 1))
    G_std = G_list[seq_idx][:, batch_idx].numpy().std(axis=(0, 1)) / np.sqrt(len(seq_idx) * bs)
    x = np.arange(G_mean.shape[0])

    # Filter: keep all for tau <= 10, every 3rd for tau > 10
    tau = x + 1
    idx_leq_10 = np.where(tau <= 10)[0]
    idx_gt_10 = np.where(tau > 10)[0]
    idx_gt_10_sparse = idx_gt_10[::2]  # every 3rd point for tau > 10
    idx_plot = np.concatenate([idx_leq_10, idx_gt_10_sparse])
    idx_plot.sort()

    color = f"C{i%10}"
    plt.errorbar(tau[idx_plot], G_mean[idx_plot], yerr=G_std[idx_plot], linestyle='', marker='o', markersize=7, label=f'{Lx}x{Ltau}', color=color, lw=2, alpha=0.8)

plt.xlabel(r"$\tau$", fontsize=17)
plt.ylabel(r"$G(\tau)$", fontsize=17)
# plt.title("G vs tau (log-log) for different lattice sizes")
# plt.legend(fontsize=10, ncol=2)
# plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')

# Manual slope and intercept for the fit line (fully manual, not normalized to data)
man_slope = -4.1
man_intercept = 10.1  # Increase this to move the fit line up
x_fit = np.arange(10, 101, dtype=float)
fit_line = np.exp(man_intercept) * x_fit ** man_slope
fit_handle, = plt.plot(
    x_fit, fit_line, 'k-', lw=1, alpha=0.8, 
    label=fr'$y \sim x^{{{man_slope:.2f}}}$', 
    zorder=100  # Ensure this line is drawn on top
)

# # Manual slope and intercept for the fit line (fully manual, not normalized to data)
# man_slope = -3
# man_intercept = 7  # Increase this to move the fit line up
# x_fit = np.arange(10, 101, dtype=float)
# fit_line = np.exp(man_intercept) * x_fit ** man_slope
# fit_handle, = plt.plot(
#     x_fit, fit_line, 'k--', lw=1, alpha=0.8, 
#     label=fr'$y \sim x^{{{man_slope:.2f}}}$', 
#     zorder=100  # Ensure this line is drawn on top
# )

import matplotlib.lines as mlines
phantom_handle = mlines.Line2D([], [], color='none', label='')
handles, labels = plt.gca().get_legend_handles_labels()
handles.insert(6, phantom_handle)
labels = [h.get_label() for h in handles]

# # Remove duplicate labels (keep order)
plt.legend(handles, labels, fontsize=15, ncol=2)

plt.ylim(10**-5, 1)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=5)))

save_dir = os.path.join(script_path, "./figures/flux_greens_loglog")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "greens_loglog_vs_tau_noncomp.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Log-log G vs tau figure saved at: {file_path}")

plt.show() 