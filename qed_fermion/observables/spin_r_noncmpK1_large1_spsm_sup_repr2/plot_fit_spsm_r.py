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

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.ticker import LogLocator

from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
set_default_plotting()
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
-8.846590453269150,
-9.20296831127156])

r_l20 = np.exp(loge_r_l20)
corr_l20 = np.exp(loge_corr_l20)

# HMC data folder
data_folder = "/Users/kx/Desktop/hmc/fignote/back_tracing/hmc_check_point_noncmpK1_large1_spsm_sup_repr2"

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
    
    plt.figure(figsize=(8, 5.8))
    main_ax = plt.gca()
    
    # Store data for normalization analysis
    all_data = {}
    # Store plot data for inset
    plot_data = []
    
    for i, Lx in enumerate(lattice_sizes):
        Ltau = int(10 * Lx)
        start = 2000 if Lx >= 20 else 4000
        sample_step = 1

        import glob
        # Find the correct file for this Lx and Ltau
        def find_hmc_file(Lx, Ltau, folder=data_folder):
            pattern = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_*_bs*_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_*_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_*_cmp_False_step_*.pt"
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
        m_nrv = re.search(r'Nrv_(\d+)', hmc_filename)
        bs = int(m_bs.group(1)) if m_bs else 1
        Nrv = int(m_nrv.group(1)) if m_nrv else 30
        # Load checkpoint data from first folder
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

        spin_corr_values = np.array(spin_corr_values)
        spin_corr_errors = np.array(spin_corr_errors)

        # Store data for inset
        plot_data.append({
            'r_values': r_values,
            'spin_corr_values': spin_corr_values,
            'spin_corr_errors': spin_corr_errors,
            'label': rf'{Ltau}x{Lx}$^2$',
            'color': color,
            'Lx': Lx
        })
        
        # Plot data and fit in log-log space (without label to remove from legend)
        main_ax.errorbar(r_values[0:], spin_corr_values[0:], 
                     yerr=np.array(spin_corr_errors[0:]),
                     linestyle=':', marker='o', color=color, 
                     markersize=12,
                     alpha=0.8)
        # plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
        #          label=f'Fit L={Lx}: y~x^{coeffs[0]:.2f}')
        

    # Plot the r_l20 and corr_l20 data on the same plot for comparison
    # plt.plot(r_l20, corr_l20, 's', color='black', label='L20 dqmc', markersize=8, alpha=0.8)
    # Linear fit for r_l20 and corr_l20 in log-log space
    log_r_l20 = np.log(r_l20)
    log_corr_l20 = np.log(corr_l20)
    coeffs_l20 = np.polyfit(log_r_l20, log_corr_l20, 1)
    r_l20_aug = np.concatenate([r_l20, [11, 13, 15, 17, 19]])
    # coeffs_l20[0] = -3.6
    # fit_line_l20 = np.exp(coeffs_l20[1] + 0.1) * r_l20_aug ** coeffs_l20[0]
    coeffs_l20[0] = -3.8
    coeffs_l20[1] = -1.99
    fit_line_l20 = np.exp(coeffs_l20[1] - 0.7) * r_l20_aug ** coeffs_l20[0]

    # Plot the fit line for L20 data
    line_fit, = main_ax.plot(r_l20_aug, fit_line_l20, 'k-', lw=1.5, alpha=0.9, label=fr'$y \sim r^{{{coeffs_l20[0]:.1f}}}$', zorder=100)

    dqmc_folder = "/Users/kx/Desktop/hmc/benchmark_dqmc/dqmc_data/kexin_benchmark_real_space_K1.0J1.25_ncomp/piflux_B0.0K1.0_largeL_tuneJ_noncompact_kexin_hk/spsm_r_odd"

    # Store DQMC data for inset
    dqmc_plot_data = []
    
    # Add dqmc data from file for L=10
    dqmc_data_path_1 = os.path.join(dqmc_folder, "l10b10js1.25jpi1.0mu0.0nf2_dqmc_bin.dat")
    dqmc_data_1 = np.loadtxt(dqmc_data_path_1)
    r_dqmc_1 = dqmc_data_1[:, 0]
    corr_dqmc_1 = dqmc_data_1[:, 1]
    err_dqmc_1 = dqmc_data_1[:, 2]
    dqmc_handle_1 = main_ax.errorbar(
        r_dqmc_1, corr_dqmc_1, yerr=err_dqmc_1, fmt='s', 
        color=f"gray", markersize=8, alpha=0.85, 
        label=fr'100x10$^2$', capsize=5,
        lw=1.2, elinewidth=2.0, capthick=2.0
    )
    dqmc_plot_data.append({
        'r_values': r_dqmc_1,
        'spin_corr_values': corr_dqmc_1,
        'spin_corr_errors': err_dqmc_1,
        'marker': 's',
        'color': 'gray',
        'label': fr'100x10$^2$'
    })

    # Add dqmc data from file for L=12
    dqmc_data_path_3 = os.path.join(dqmc_folder, "l12b12js1.25jpi1.0mu0.0nf2_dqmc_bin.dat")
    dqmc_data_3 = np.loadtxt(dqmc_data_path_3)
    r_dqmc_3 = dqmc_data_3[:, 0]
    corr_dqmc_3 = dqmc_data_3[:, 1]
    err_dqmc_3 = dqmc_data_3[:, 2]
    dqmc_handle_3 = main_ax.errorbar(
        r_dqmc_3, corr_dqmc_3, yerr=err_dqmc_3, fmt='^', 
        color="gray", markersize=8, alpha=0.85, 
        label=fr'120x12$^2$', capsize=5, 
        lw=1.2, elinewidth=2.0, capthick=2.0
    )
    dqmc_plot_data.append({
        'r_values': r_dqmc_3,
        'spin_corr_values': corr_dqmc_3,
        'spin_corr_errors': err_dqmc_3,
        'marker': '^',
        'color': 'gray',
        'label': fr'120x12$^2$'
    })

    # # Add dqmc data from file for L=16
    dqmc_data_path_2 = os.path.join(dqmc_folder, "l16b16js1.25jpi1.0mu0.0nf2_dqmc_bin.dat")
    dqmc_data_2 = np.loadtxt(dqmc_data_path_2)
    r_dqmc_2 = dqmc_data_2[:, 0]
    corr_dqmc_2 = dqmc_data_2[:, 1]
    err_dqmc_2 = dqmc_data_2[:, 2]
    dqmc_handle_2 = main_ax.errorbar(
        r_dqmc_2, corr_dqmc_2, yerr=err_dqmc_2, fmt='D', 
        color=f"gray", markersize=7, alpha=0.85, 
        label=rf'160x16$^2$', capsize=5,
        lw=1.2, elinewidth=2.0, capthick=2.0
    )
    dqmc_plot_data.append({
        'r_values': r_dqmc_2,
        'spin_corr_values': corr_dqmc_2,
        'spin_corr_errors': err_dqmc_2,
        'marker': 'D',
        'color': 'gray',
        'label': rf'160x16$^2$'
    })

    # Build legend with only fit line and DQMC data (no colored HMC points)
    handles = [line_fit, dqmc_handle_1, dqmc_handle_3, dqmc_handle_2]
    handles = handles[1:] + handles[0:1]
    # phantom
    # handles.extend([mlines.Line2D([], [], color='none', label='') for _ in range(6)])
    # handles = handles[:10] + handles[14:] + handles[10:14]

    labels = [line.get_label() for line in handles]
    # Linear axes
    main_ax.set_xlabel('r', fontsize=23)
    main_ax.set_ylabel(r'$C_S^{\uparrow\downarrow}(r, 0)$', fontsize=23)

    # Move the legend a little lower in the upper left by adjusting bbox_to_anchor
    main_ax.legend(handles, labels, ncol=1, fontsize=18, loc="upper left", bbox_to_anchor=(0.00001, 0.93))
    main_ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Set log scales
    main_ax.set_xscale('log')
    main_ax.set_yscale('log')

    # set tick label size
    main_ax.xaxis.set_tick_params(labelsize=22)
    main_ax.yaxis.set_tick_params(labelsize=22)

    # Turn off minor ticks on both axes
    main_ax.yaxis.set_minor_locator(plt.NullLocator())

    # Set y-axis lower limit to 1e-7
    main_ax.set_ylim(10**(-8.5), 10**-1.0)
    main_ax.set_xlim(0.2, None)
    
    # --- Inset axes ---
    # Determine inset limits based on grey DQMC data region
    # Find the r range and corr range for DQMC data
    all_dqmc_r = np.concatenate([r_dqmc_1, r_dqmc_3, r_dqmc_2])
    all_dqmc_corr = np.concatenate([corr_dqmc_1, corr_dqmc_3, corr_dqmc_2])
    all_dqmc_err = np.concatenate([err_dqmc_1, err_dqmc_3, err_dqmc_2])
    
    # Set inset limits to zoom into grey points region
    inset_width = 0.40
    inset_height = 0.40
    inset_ax = inset_axes(
        main_ax,
        width="100%", height="100%",
        loc='lower left',
        bbox_to_anchor=(0.09, 0.09, inset_width, inset_height),
        bbox_transform=main_ax.transAxes,
        borderpad=0
    )
    
    # Plot all HMC data in the inset (filter by r range and L <= 160, i.e., Lx <= 16)
    for entry in plot_data:
        # Only plot data up to L=160 (Ltau=160 corresponds to Lx=16)
        if entry['Lx'] > 16:
            continue
        r_vals = np.array(entry['r_values'])
        corr_vals = np.array(entry['spin_corr_values'])
        corr_errs = np.array(entry['spin_corr_errors'])
        # Filter data points within inset x range
        inset_ax.errorbar(r_vals, corr_vals, 
                            yerr=corr_errs,
                            linestyle=':', marker='o', 
                            color=entry['color'], 
                            markersize=8, alpha=0.8)
    
    # Plot all DQMC data in the inset
    for dqmc_entry in dqmc_plot_data:
        r_vals = dqmc_entry['r_values']
        corr_vals = dqmc_entry['spin_corr_values']
        corr_errs = dqmc_entry['spin_corr_errors']
        # Filter data points within inset x range
        inset_ax.errorbar(r_vals, corr_vals, 
                            yerr=corr_errs,
                            fmt=dqmc_entry['marker'],
                            color=dqmc_entry['color'],
                            markersize=6, alpha=0.85, capsize=5 ,
                            lw=1.2, elinewidth=2.0, capthick=2.0)

    # Set inset limits and scales
    inset_xlim = (4.2, 8.0)
    inset_ylim = (2e-5, 0.0008)
    inset_ax.set_xlim(*inset_xlim)
    inset_ax.set_ylim(*inset_ylim)
    inset_ax.set_xscale('log')
    inset_ax.set_yscale('log')
    inset_ax.set_xlabel("", fontsize=12)
    inset_ax.set_ylabel("", fontsize=12)

    # Remove minor ticks in inset
    inset_ax.xaxis.set_minor_locator(plt.NullLocator())
    inset_ax.yaxis.set_minor_locator(plt.NullLocator())
    # Set x-ticks for inset
    from matplotlib.ticker import FixedLocator
    inset_xticks = [5, 6, 7, 8]
    inset_ax.set_xticks(inset_xticks)
    inset_ax.xaxis.set_major_locator(FixedLocator(inset_xticks))
    # xtick_labels = [item.get_text() for item in inset_ax.get_xticklabels()]
    # Only keep the x ticklabels for 5 and 8 in scientific notation (keep as 5 and 8 but formatted like previous 5e0)
    xtick_labels = []
    for tick in inset_xticks:
        if tick in [5, 6, 7, 8]:
            xtick_labels.append(r"${}$".format(tick))
        else:
            xtick_labels.append('')
    inset_ax.set_xticklabels(xtick_labels)

    # Set tick label fontsize for both axes to match
    inset_ax.xaxis.set_tick_params(labelsize=16)
    inset_ax.yaxis.set_tick_params(labelsize=16)
    # Set shorter tick length for inset axes ticks
    inset_ax.tick_params(axis='both', which='both', length=4)

    # Rectangle on main plot to show inset region
    rect = mpatches.Rectangle((inset_xlim[0], inset_ylim[0]), 
                             inset_xlim[1]-inset_xlim[0], 
                             inset_ylim[1]-inset_ylim[0], 
                             linewidth=1.0, edgecolor=inset_ax.spines['bottom'].get_edgecolor(), 
                             linestyle='--', facecolor='none', zorder=200)
    main_ax.add_patch(rect)

    # Save the plot (log-log axes)
    save_dir = os.path.join(script_path, f"./figures/spin_r_fit_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, "spin_r_vs_x_fit_log_noncmpK1_tmp.pdf")
    png_path = os.path.join(save_dir, "spin_r_vs_x_fit_log_noncmpK1_tmp.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Log-log figure saved at: {pdf_path}")

    plt.show()

 
if __name__ == '__main__':
    plot_spin_r()
    
    dbstop = 1

    # [1:] *= 1.25
    # [2:] *= 1.6


