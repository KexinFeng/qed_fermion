import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np
import os
import torch
import sys
import glob

plt.ion()
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
set_default_plotting()

import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Parameters for data loading and plotting
start = 6000  # Skip initial equilibration steps
sample_step = 1

# Lattice sizes to analyze
lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56]

# Data folder (same as in plot_fit_spsm_r.py)
hmc_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmpK0_large4_BBr/hmc_check_point_noncmpK0_large4_BBr"

plt.figure(figsize=(8, 6))
main_ax = plt.gca()

# Store processed data for reuse in both main and inset plots
plot_data = []

for i, Lx in enumerate(lattice_sizes):
    Ltau = int(10 * Lx)
    Ly = Lx
    
    # Use glob to find the correct file for this Lx and Ltau
    def find_hmc_file(Lx, Ltau):
        pattern = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_*_bs*_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_*_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5*_cmp_False_step_*.pt"
        files = glob.glob(os.path.join(hmc_folder, pattern))
        if not files:
            print(f"No file found for Lx={Lx}, Ltau={Ltau}")
            return None
        # Pick the file with the largest step (sort by step number)
        def extract_step(filename):
            m = re.search(r'step_(\d+)\.pt', filename)
            return int(m.group(1)) if m else 0
        files.sort(key=extract_step, reverse=True)
        return files[0]

    hmc_filename = find_hmc_file(Lx, Ltau)
    if hmc_filename is None:
        continue

    # Parse bs from filename
    m_bs = re.search(r'bs(\d+)', hmc_filename)
    bs = int(m_bs.group(1)) if m_bs else 1

    res = torch.load(hmc_filename, map_location='cpu')
    print(f'Loaded: {hmc_filename}')

    G_list = res['G_list']
    hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    end = int(hmc_match.group(1))
    # bs is already parsed above

    seq_idx = np.arange(start, end, sample_step)
    batch_idx = np.arange(bs)

    G_mean = G_list[seq_idx][:, batch_idx].numpy().mean(axis=(0, 1))
    G_std = G_list[seq_idx][:, batch_idx].numpy().std(axis=(0, 1)) / np.sqrt(len(seq_idx) * bs)
    x = np.arange(G_mean.shape[0])
    tau = x + 1

    idx_leq_10 = np.where(tau <= 10)[0]
    idx_gt_10 = np.where(tau > 10)[0]
    idx_gt_10_sparse = idx_gt_10[::2]  # every 3rd point for tau > 10
    idx_plot = np.concatenate([idx_leq_10, idx_gt_10_sparse])
    idx_plot.sort()

    color = f"C{i%10}"
    # Store all relevant data for later plotting
    plot_data.append({
        'tau': tau,
        'G_mean': G_mean,
        'G_std': G_std,
        'idx_plot': idx_plot,
        'label': f'{Lx}x{Ltau}',
        'color': color
    })
    # Plot on main axis
    main_ax.errorbar(tau[idx_plot], G_mean[idx_plot], yerr=G_std[idx_plot], linestyle='', marker='o', markersize=7, label=f'{Lx}x{Ltau}', color=color, lw=2, alpha=0.8)

main_ax.set_xlabel(r"$\tau$", fontsize=17)
main_ax.set_ylabel(r"$G(\tau)$", fontsize=17)
plt.tight_layout()
main_ax.set_xscale('log')
main_ax.set_yscale('log')

# Manual slope and intercept for the fit line (fully manual, not normalized to data)
man_slope = -4.0
man_intercept = 10.4  # Increase this to move the fit line up
x_fit = np.arange(10, 101, dtype=float)
fit_line = np.exp(man_intercept) * x_fit ** man_slope
fit_handle, = main_ax.plot(
    x_fit, fit_line, 'k-', lw=1, alpha=0.8, 
    label=fr'$y \sim x^{{{man_slope:.2f}}}$', 
    zorder=100  # Ensure this line is drawn on top
)

import matplotlib.lines as mlines
phantom_handle = mlines.Line2D([], [], color='none', label='')
handles, labels = main_ax.get_legend_handles_labels()
handles.insert(6, phantom_handle)
labels = [h.get_label() for h in handles]
main_ax.legend(handles, labels, fontsize=15, ncol=2)

main_ax.set_ylim(10**-5, 1)
ax = main_ax
ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=8)))

# --- Inset axes ---
# Place inset at top left, slightly below y=4e-1
inset_width = 0.33  # fraction of main_ax width (smaller)
inset_height = 0.33  # fraction of main_ax height (smaller)
# Use axes coordinates for precise placement
# bbox_to_anchor must be a 4-tuple (x0, y0, width, height) in axes fraction

inset_ax = inset_axes(
    main_ax,
    width="100%", height="100%",
    loc='upper left',
    bbox_to_anchor=(0.08, 0.47, inset_width, inset_height),  # x0, y0, width, height (moved right)
    bbox_transform=main_ax.transAxes,
    borderpad=0
)

# Plot the stored data in the inset
for entry in plot_data:
    inset_ax.errorbar(entry['tau'][entry['idx_plot']], entry['G_mean'][entry['idx_plot']], yerr=entry['G_std'][entry['idx_plot']], linestyle='', marker='o', markersize=5, color=entry['color'], lw=1, alpha=0.8)

# Fit line in inset
inset_ax.plot(x_fit, fit_line, 'k-', lw=1, alpha=0.8, zorder=100)

# Set new xlim for inset
inset_xlim = (20, 90)
inset_ylim = (2e-3, 9e-2)
inset_ax.set_xlim(*inset_xlim)
inset_ax.set_ylim(*inset_ylim)
inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
inset_ax.tick_params(axis='both', which='major', labelsize=10)
# Set x-ticks and formatter for inset
inset_xticks = [20, 30, 40, 50, 60, 70]
inset_ax.set_xticks(inset_xticks)
inset_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x)) if x in inset_xticks else ""))
inset_ax.set_yticks([2e-3, 1e-2])
# inset_ax.set_yticks([])
inset_ax.set_xlabel("", fontsize=10)
inset_ax.set_ylabel("", fontsize=10)

# Rectangle on main plot to show inset region
# Rectangle parameters updated to match new inset limits
rect = mpatches.Rectangle((inset_xlim[0], inset_ylim[0]), inset_xlim[1]-inset_xlim[0], inset_ylim[1]-inset_ylim[0], linewidth=1.5, edgecolor='k', linestyle='--', facecolor='none', zorder=200)
main_ax.add_patch(rect)

save_dir = os.path.join(script_path, "./figures/flux_greens_loglog")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "greens_loglog_vs_tau_noncomp.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Log-log G vs tau figure saved at: {file_path}")

plt.show() 