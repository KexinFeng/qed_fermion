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

from qed_fermion.utils.prep_plots import set_default_plotting, selective_log_label_func

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

# HMC data folder for large4_BBr
data_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmpK0_large4_BBr/hmc_check_point_noncmpK0_large4_BBr"

# Set default plotting settings for physics scientific publication (Matlab style)
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()  

def plot_spin_r():
    """Plot spin-spin correlation as a function of distance r for different lattice sizes (large4_BBr)."""
    
    # Define lattice sizes to analyze (from data directory)
    lattice_sizes = [12, 16, 20, 30, 36, 40]
    
    # Sampling parameters
    start = 6000  # Skip initial equilibration steps
    sample_step = 1
    
    plt.figure(figsize=(8, 6))
    
    # Store data for normalization analysis
    all_data = {}
    
    for i, Lx in enumerate(lattice_sizes):
        # Construct filename for this lattice size
        Ltau = int(10 * Lx)
        if Lx in [12, 16]:
            Nrv = 30
            bs = 2
        elif Lx in [20, 30, 36]:
            Nrv = 40
            bs = 2
        elif Lx == 40:
            Nrv = 40
            bs = 1
        else:
            raise ValueError(f"Unexpected Lx: {Lx}")
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs{bs}_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_{Nrv}_cmp_False_step_10000.pt"
        hmc_filename = os.path.join(data_folder, hmc_file)
        
        if not os.path.exists(hmc_filename):
            raise FileNotFoundError(f"File not found: {hmc_filename}")
            
        # Load checkpoint data
        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')
        
        # Extract spin-spin correlation data: BB_r_list
        bb_r = res['BB_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]
        
        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)
        hmc_match_bs = re.search(r'bs(\d+)', hmc_filename)
        bs = int(hmc_match_bs.group(1))

        # Average over equilibrated timesteps and batch dimension
        bb_r_avg = bb_r[seq_idx].mean(dim=(0, 1))       # Average over [timesteps, batches] -> [Ly, Lx]
        bb_r_avg_std = bb_r[seq_idx].std(dim=(0, 1))       # Average over [timesteps, batches] -> [Ly, Lx]
        bb_r_avg_abs = bb_r_avg.abs()            # Take absolute value for correlation
        
        # Convert to numpy for easier manipulation
        bb_r_np = bb_r_avg.numpy()
        bb_r_np_abs = bb_r_avg_abs.numpy()
        bb_r_avg_std_np = bb_r_avg_std.numpy()
        
        r_values = []
        spin_corr_values = []
        spin_corr_errors = []
        
        # Simplified: plot spin correlation along x-direction only (y=0)
        for r in range(start_dist, Lx, step_dist):
            x = r
            y = y_diplacement(x) 
            
            r_values.append(r)
            val = 1/2 * (bb_r_np_abs[y, x] + bb_r_np_abs[y, Lx - x]) if y != x else bb_r_np_abs[y, x]
            err = 1/2 * (bb_r_avg_std_np[y, x] + bb_r_avg_std_np[y, Lx - x]) if y != x else bb_r_avg_std_np[y, x] 
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

        # Linear fit in log-log space
        log_r = np.log(r_fit)
        log_corr = np.log(spin_corr_fit)
        coeffs = np.polyfit(log_r, log_corr, 1)
        fit_line = np.exp(coeffs[1]) * r_fit**coeffs[0]

        # Plot data and fit in log-log space
        plt.errorbar(r_values[0:], spin_corr_values[0:], yerr=spin_corr_errors[0:], 
                     linestyle=':', marker='o', color=color, 
                     label=f'{Lx}x{Ltau}', alpha=0.8)
        # plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
        #          label=f'Fit L={Lx}: y~x^{coeffs[0]:.2f}')
        
    # Linear axes
    plt.xlabel('r', fontsize=19)
    plt.ylabel('$\\langle S^{+}_r S^{-}_0 \\rangle$', fontsize=19)

    # Add a reference fit line with coeff[0] = -3.3 and coeff[1] = 0
    r_min = min([min(d['r_values']) for d in all_data.values() if d['r_values']])
    r_max = max([max(d['r_values']) for d in all_data.values() if d['r_values']])
    r_fitline = np.linspace(r_min, (r_max + r_min + 2)// 2, 100)
    coeff0 = -3.5
    coeff1 = -3.5
    fit_line = np.exp(coeff1) * r_fitline ** coeff0
    handles, labels = plt.gca().get_legend_handles_labels()
    line_fit, = plt.plot(r_fitline, fit_line, 'k-', lw=1., alpha=0.9, label=fr'$y \sim x^{{{coeff0:.2f}}}$')
    handles.insert(0, line_fit)
    plt.legend(handles, [line.get_label() for line in handles], ncol=2)

    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Set y-axis lower limit to 1e-7
    plt.ylim(1e-7, None)
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=6)))

    # Save the plot (log-log axes)
    save_dir = os.path.join(script_path, f"./figures/spin_r_fit_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "spin_r_vs_x_fit_log_noncmpK0_large4_BBr.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Log-log figure saved at: {file_path}")

    plt.show()

if __name__ == '__main__':
    plot_spin_r() 