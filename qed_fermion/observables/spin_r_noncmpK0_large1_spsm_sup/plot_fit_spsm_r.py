import re
import matplotlib.pyplot as plt

plt.ion()

import numpy as np

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.ticker import LogLocator

from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
set_default_plotting()
import matplotlib.lines as mlines

start_dist = 1
step_dist = 2 
y_diplacement = lambda x: 0

# Only use r > 0 for log-log fit to avoid log(0)
lw = 3 if start_dist == 0 else 0
up = 7

suffix = None
if start_dist == 1 and step_dist == 2:
    suffix = "odd"
elif start_dist in {0, 2} and step_dist == 2:
    suffix = "even"
else:
    if y_diplacement(1) == 0:
        suffix = "all"
    else:
        suffix = "diag"

# Data
loge_r_l20 = np.array([-0.000672932415154438,
1.1003491771532000,
1.6095347046200500,
1.9497992418294800,
2.202148897512390])

loge_corr_l20 = np.array([-2.8412555154432400,
-6.0591656638588000,
-7.766746891295630,
-8.746590453269150,
-9.20296831127156])

r_l20 = np.exp(loge_r_l20)
corr_l20 = np.exp(loge_corr_l20)

# HMC data folder
hmc_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmpK0_large1_spsm/hmc_check_point_noncmpK0_large1_spsm"

# Set default plotting settings for physics scientific publication (Matlab style)
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()  

def plot_spin_r():
    """Plot spin-spin correlation as a function of distance r for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [6, 8, 10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    lattice_sizes = [8, 10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    # lattice_sizes = [8, 12, 16, 20, 30, 40, 56, 60]
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
     
    # Sampling parameters
    start = 6000  # Skip initial equilibration steps
    sample_step = 1
    
    plt.figure(figsize=(8, 6))
    
    # Store data for normalization analysis
    all_data = {}
    
    for i, Lx in enumerate(lattice_sizes):
        # Construct filename for this lattice size
        Ltau = int(10 * Lx)
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

        hmc_filename = os.path.join(hmc_folder, hmc_file)
        
        if not os.path.exists(hmc_filename):
            raise FileNotFoundError(f"File not found: {hmc_filename}")
            
        # Load checkpoint data
        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')
        
        # Extract spin-spin correlation data: spsm_r_list
        spsm_r = res['spsm_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]
        
        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)
        hmc_match_bs = re.search(r'bs(\d+)', hmc_filename)
        bs = int(hmc_match_bs.group(1))

        # Average over equilibrated timesteps and batch dimension
        spsm_r_avg = spsm_r[seq_idx].mean(dim=(0, 1))       # Average over [timesteps, batches] -> [Ly, Lx]
        spsm_r_avg_std = spsm_r[seq_idx].std(dim=(0, 1))       # Average over [timesteps, batches] -> [Ly, Lx]
        spsm_r_avg_abs = spsm_r_avg.abs()            # Take absolute value for correlation
        
        # Convert to numpy for easier manipulation
        spsm_r_np = spsm_r_avg.numpy()
        spsm_r_np_abs = spsm_r_avg_abs.numpy()
        spsm_r_avg_std_np = spsm_r_avg_std.numpy()
        
        r_values = []
        spin_corr_values = []
        spin_corr_errors = []
        
        # Simplified: plot spin correlation along x-direction only (y=0)
        for r in range(start_dist, Lx, step_dist):
            x = r
            y = y_diplacement(x) 
            
            r_values.append(r)
            val = 1/2 * (spsm_r_np_abs[y, x] + spsm_r_np_abs[y, Lx - x]) if y != x else spsm_r_np_abs[y, x]
            err = 1/2 * (spsm_r_avg_std_np[y, x] + spsm_r_avg_std_np[y, Lx - x]) if y != x else spsm_r_avg_std_np[y, x] 
            spin_corr_values.append(val)
            spin_corr_errors.append(err / np.sqrt(len(seq_idx) * bs))

        # Store data for analysis
        all_data[Lx] = {
            'r_values': r_values,
            'spin_corr_values': spin_corr_values,
            'spin_corr_errors': spin_corr_errors,
            'normalization': spin_corr_values[0] if spin_corr_values else 1.0  # r=0 value for normalization
        }
        
        # Plot spin correlation vs distance for this lattice size (log-log with linear fit)
        color = f"C{i}"
        # Only use r > 0 for log-log fit to avoid log(0)
        r_fit = np.array(r_values[lw:up])
        spin_corr_fit = np.array(spin_corr_values[lw:up])
        # spin_corr_err_fit = np.array(spin_corr_errors[lw:up])

        # Linear fit in log-log space
        log_r = np.log(r_fit)
        log_corr = np.log(spin_corr_fit)
        coeffs = np.polyfit(log_r, log_corr, 1)
        fit_line = np.exp(coeffs[1]) * r_fit**coeffs[0]

        # Plot data and fit in log-log space
        plt.errorbar(r_values[0:], spin_corr_values[0:], yerr=spin_corr_errors[0:], 
                     linestyle=':', marker='o', color=color, 
                     label=fr'${Ltau}\times {Lx}^2$', alpha=0.8)
        # plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
        #          label=f'Fit L={Lx}: y~x^{coeffs[0]:.2f}')
        
        dbstop = 1
    
    # Plot the r_l20 and corr_l20 data on the same plot for comparison
    # plt.plot(r_l20, corr_l20, 's', color='black', label='L20 dqmc', markersize=8, alpha=0.8)
    # Linear fit for r_l20 and corr_l20 in log-log space
    log_r_l20 = np.log(r_l20)
    log_corr_l20 = np.log(corr_l20)
    coeffs_l20 = np.polyfit(log_r_l20, log_corr_l20, 1)
    r_l20_aug = np.concatenate([r_l20, [11, 13, 15, 17, 19]])
    # coeffs_l20[0] = -3.6
    # fit_line_l20 = np.exp(coeffs_l20[1] + 0.1) * r_l20_aug ** coeffs_l20[0]
    coeffs_l20[0] = -3.3
    fit_line_l20 = np.exp(coeffs_l20[1] - 0.7) * r_l20_aug ** coeffs_l20[0]

    # Plot the fit line and merge its handle/label to the end of the existing legend entries
    handles, _ = plt.gca().get_legend_handles_labels()
    
    # Plot the fit line for L20 data
    line_fit, = plt.plot(r_l20_aug, fit_line_l20, 'k-', lw=1., alpha=0.9, label=fr'$y \sim r^{{{coeffs_l20[0]:.2f}}}$')
    # Ensure the fit line is appended at the end
    handles.insert(0, line_fit)
    place_holder_handle = mlines.Line2D([], [], color='none', label='')
    handles.insert(6, place_holder_handle)
    labels = [line.get_label() for line in handles]

    # Linear axes
    plt.xlabel('r', fontsize=19)
    plt.ylabel(r'$C_S^{\uparrow\downarrow}(r, 0)$', fontsize=19)

    plt.legend(handles, labels, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=6)))

    # Set y-axis lower limit to 1e-7
    plt.ylim(1e-7, None)

    # Save the plot (log-log axes)
    save_dir = os.path.join(script_path, f"./figures/spin_r_fit_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "spin_r_vs_x_fit_log_noncmpK0.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Log-log figure saved at: {file_path}")

    plt.show()

 

if __name__ == '__main__':
    plot_spin_r()
    
    dbstop = 1


