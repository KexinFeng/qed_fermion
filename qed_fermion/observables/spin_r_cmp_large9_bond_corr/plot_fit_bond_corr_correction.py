import re
import matplotlib.pyplot as plt
import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')

from qed_fermion.utils.util import convolution

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
data_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/bond_corr/hmc_check_point_cmp_large10_bond_corr_part2"
data_folder2 = ""

separate = False

# Set default plotting settings for physics scientific publication (Matlab style)
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()  

def plot_spin_r():
    """Plot spin-spin correlation as a function of distance r for different lattice sizes (large4_BBr)."""
    
    # Define lattice sizes to analyze (from data directory)
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    lattice_sizes = [12, 16, 20, 30, 36, 40, 46, 56, 60]
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Sampling parameters
    # start = 5000  # Skip initial equilibration steps
    sample_step = 1
    
    plt.figure(figsize=(8, 6))
    
    # Store data for normalization analysis
    all_data = {}
    
    for i, Lx in enumerate(lattice_sizes):
        # Construct filename for this lattice size
        Ltau = int(10 * Lx)
        start = 2000

        import glob
        # Find the correct file for this Lx and Ltau
        def find_hmc_file(Lx, Ltau, folder=data_folder):
            pattern = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_*_bs*_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_*_cmp_True_step_*.pt"
            files = glob.glob(os.path.join(folder, pattern))
            if not files:
                print(f"No file found for Lx={Lx}, Ltau={Ltau} in {folder}")
                return None
            # Pick the file with the largest step (sort by step number)
            def extract_step(filename):
                m = re.search(r'step_(\d+)\\.pt', filename)
                return int(m.group(1)) if m else 0
            files.sort(key=extract_step, reverse=True)
            return files[0]

        # Load data from first folder
        hmc_filename = find_hmc_file(Lx, Ltau)
        if hmc_filename is None:
            continue
        # Now parse bs and Nrv from filename
        m_bs = re.search(r'bs(\d+)', hmc_filename)
        bs = int(m_bs.group(1)) if m_bs else 1

        # Load checkpoint data from first folder
        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')
        
        # Extract spin-spin correlation data: BB_r_list
        bb_r = res['BB_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]
        bb0_r = res['BB0_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]
        b_r = res['B_r_list']
        
        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'step_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)

        # Average over equilibrated timesteps and batch dimension for first folder
        bb0_r_avg1 = bb0_r[seq_idx].mean(dim=(0, 1))       # Average over [timesteps, batches] -> [Ly, Lx]
        v_r_seq1 = b_r[seq_idx]
        v_r_avg1 = v_r_seq1.mean(dim=(0, 1), keepdim=True)       # Average over [timesteps, batches] -> [Ly, Lx]
        bb0_r_avg_std1 = bb0_r[seq_idx].std(dim=(0, 1))       # Average over [timesteps, batches] -> [Ly, Lx]

        vv_r_seq1 = convolution(v_r_seq1)
        vv_r_avg1 = convolution(v_r_avg1)
        bb1_r_avg1 = (4 * (vv_r_seq1 - vv_r_avg1)).mean(dim=(0, 1))
        bb1_r_avg_std1 = (4 * (vv_r_seq1 - vv_r_avg1)).std(dim=(0, 1))

        # Convert to numpy for easier manipulation
        bb_r_np_abs1 = bb1_r_avg1.abs().numpy()
        bb0_r_np_abs1 = bb0_r_avg1.abs().numpy()
        bb_r_avg_std_np1 = bb1_r_avg_std1.numpy()
        bb0_r_avg_std_np1 = bb0_r_avg_std1.numpy()

        # Load data from second folder
        hmc_filename2 = find_hmc_file(Lx, Ltau, data_folder2)
        if hmc_filename2 is None:
            # If no file found in second folder, use only first folder data
            bb_r_np_abs = bb_r_np_abs1
            bb0_r_np_abs = bb0_r_np_abs1
            bb_r_avg_std_np = bb_r_avg_std_np1
            bb0_r_avg_std_np = bb0_r_avg_std_np1
            total_samples = len(seq_idx) * bs
        else:
            # Load checkpoint data from second folder
            res2 = torch.load(hmc_filename2, map_location='cpu')
            print(f'Loaded: {hmc_filename2}')
            
            # Extract required lists from second folder
            bb0_r2 = res2['BB0_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]
            b_r2 = res2['B_r_list']
            
            # Extract sequence indices for equilibrated samples
            # hmc_match2 = re.search(r'Nstp_(\d+)', hmc_filename2)
            # end2 = int(hmc_match2.group(1))
            start2 = 0
            end2 = 4000
            seq_idx2 = np.arange(start2, end2, sample_step)

            # Average over equilibrated timesteps and batch dimension for second folder (newest calc)
            bb0_r_avg2 = bb0_r2[seq_idx2].mean(dim=(0, 1))
            v_r_seq2 = b_r2[seq_idx2]
            v_r_avg2 = v_r_seq2.mean(dim=(0, 1), keepdim=True)
            bb0_r_avg_std2 = bb0_r2[seq_idx2].std(dim=(0, 1))

            vv_r_seq2 = convolution(v_r_seq2)
            vv_r_avg2 = convolution(v_r_avg2)
            bb1_r_avg2 = (4 * (vv_r_seq2 - vv_r_avg2)).mean(dim=(0, 1))
            bb1_r_avg_std2 = (4 * (vv_r_seq2 - vv_r_avg2)).std(dim=(0, 1))
            
            # Convert to numpy for easier manipulation
            bb_r_np_abs2 = bb1_r_avg2.abs().numpy()
            bb0_r_np_abs2 = bb0_r_avg2.abs().numpy()
            bb_r_avg_std_np2 = bb1_r_avg_std2.numpy()
            bb0_r_avg_std_np2 = bb0_r_avg_std2.numpy()

            # Combine data from both folders (average)
            n1 = len(seq_idx) * bs
            n2 = len(seq_idx2) * bs
            total_samples = n1 + n2
            bb_r_np_abs = (bb_r_np_abs1 * n1 + bb_r_np_abs2 * n2) / total_samples
            bb0_r_np_abs = (bb0_r_np_abs1 * n1 + bb0_r_np_abs2 * n2) / total_samples
            # For the standard deviation, combine variances weighted by sample size, then take sqrt
            bb_r_avg_std_np = np.sqrt(
                (bb_r_avg_std_np1**2 * n1 + bb_r_avg_std_np2**2 * n2) / total_samples
            )
            bb0_r_avg_std_np = np.sqrt(
                (bb0_r_avg_std_np1**2 * n1 + bb0_r_avg_std_np2**2 * n2) / total_samples
            )
        
        r_values = []
        spin_corr_values = []
        spin_corr_values0 = []
        spin_corr_errors = []
        spin_corr_errors0 = []
        
        # Simplified: plot spin correlation along x-direction only (y=0)
        for r in range(start_dist, Lx, step_dist):
            x = r
            y = y_diplacement(x) 
            
            r_values.append(r)
            val = 1/2 * (bb_r_np_abs[y, x] + bb_r_np_abs[y, Lx - x]) if y != x else bb_r_np_abs[y, x]  # bb_r_np_abs[Ly - y, x] will err, since y = 0.
            val0 = 1/2 * (bb0_r_np_abs[y, x] + bb0_r_np_abs[y, Lx - x]) if y != x else bb0_r_np_abs[y, x]  # bb_r_np_abs[Ly - y, x] will err, since y = 0.
            err = 1/2 * (bb_r_avg_std_np[y, x] + bb_r_avg_std_np[y, Lx - x]) if y != x else bb_r_avg_std_np[y, x] 
            err0 = 1/2 * (bb0_r_avg_std_np[y, x] + bb0_r_avg_std_np[y, Lx - x]) if y != x else bb0_r_avg_std_np[y, x] 

            spin_corr_values.append(val)
            spin_corr_values0.append(val0)
            spin_corr_errors.append(err / np.sqrt(total_samples))
            spin_corr_errors0.append(err0 / np.sqrt(total_samples))

        spin_corr_values = np.array(spin_corr_values)
        spin_corr_values0 = np.array(spin_corr_values0)
        spin_corr_errors = np.array(spin_corr_errors)
        spin_corr_errors0 = np.array(spin_corr_errors0)
        # spin_corr_values = spin_corr_values - spin_corr_values.min() + 1e-15
        spin_corr_values = spin_corr_values

        # Store data for analysis
        all_data[Lx] = {
            'r_values': r_values,
            'spin_corr_values': spin_corr_values,
            'spin_corr_errors': spin_corr_errors,
            'normalization': spin_corr_values[0] if spin_corr_values is not None else 1.0  # r=0 value for normalization
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
        # Plot error bars with alpha=1 (fully opaque)
        if separate:
            eb = plt.errorbar(r_values[0:], spin_corr_values[0:], yerr=spin_corr_errors[0:], 
                            linestyle=':', marker='o', color=color, 
                            label=rf'${Ltau}x{Lx}^2$_corr(vivj)', alpha=0.8)
            eb2 = plt.errorbar(r_values[0:], spin_corr_values0[0:], yerr=spin_corr_errors0[0:], 
                            linestyle='-', marker='^', color=color, 
                            label=rf'${Ltau}x{Lx}^2$_original', alpha=0.8)
            if hasattr(eb2, 'lines') and len(eb2.lines) > 0:
                eb2.lines[0].set_alpha(0.8)
        else:
            eb = plt.errorbar(r_values[0:], 
                            (spin_corr_values + spin_corr_values0)[0:], 
                            yerr=(((spin_corr_errors**2 + spin_corr_errors0**2) /2 )**(1/2))[0:], 
                            linestyle=':', marker='o', color=color, 
                            label=rf'${Ltau}x{Lx}^2$', alpha=1.0)        
        
        # Set only the marker (dots) to have alpha=0.8
        if hasattr(eb, 'lines') and len(eb.lines) > 0:
            eb.lines[0].set_alpha(0.8)

        dbstop = 1

    # Linear axes
    plt.xlabel('r', fontsize=19)
    plt.ylabel('$C_B(r, 0)$', fontsize=19)

    # Add a reference fit line with coeff[0] = -3.3 and coeff[1] = 0
    r_min = min([min(d['r_values']) for d in all_data.values() if d['r_values']])
    r_max = max([max(d['r_values']) for d in all_data.values() if d['r_values']])
    r_fitline = np.linspace(r_min - 0.4, (r_max + r_min - 6)// 2, 100)
    coeff0 = -3.2
    coeff1 = -3.3
    fit_line = np.exp(coeff1) * r_fitline ** coeff0
    handles, labels = plt.gca().get_legend_handles_labels()
    line_fit, = plt.plot(r_fitline, fit_line, 'k-', lw=1., alpha=0.9, label=fr'$y \sim r^{{{coeff0:.1f}}}$', zorder=100)
    handles.insert(0, line_fit)

    # phantom
    phantom_line = mlines.Line2D([], [], color='none', label='')
    handles.insert(len(handles) // 2 + 1, phantom_line)

    # Ensure the fit line is appended at the end
    # place_holder_handle = mlines.Line2D([], [], color='none', label='')
    # handles.insert(5, place_holder_handle)
    labels = [line.get_label() for line in handles]
    plt.legend(handles, labels, ncol=2, fontsize=13 if not separate else 8)

    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    # Set y-axis lower limit to 1e-7
    # plt.ylim(1e-6, 10**-0.5)
    if not separate:
        plt.ylim(10**-6.5, 10**-0.8)
        plt.xlim(0.85, None)
    else:
        plt.ylim(10**-10, 10**-0.5)
        plt.xlim(0.5, None)     

    ax = plt.gca()
    # ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=6)))

    # Save the plot (log-log axes)
    save_dir = os.path.join(script_path, f"./figures/BB_r_fit_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 
                             ("sep_" if separate else "") + "BB_r_vs_x_fit_log_cmp_large8_BBr.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Log-log figure saved at: {file_path}")

    plt.show()

if __name__ == '__main__':
    plot_spin_r() 