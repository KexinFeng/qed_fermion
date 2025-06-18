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
hmc_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/cmp_large3_asym4/hmc_check_point_cmp_large3_asym4"
   
def plot_spin_r():
    """Plot spin-spin correlation as a function of distance r for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [6, 8, 10, 12, 16, 20, 24]
     
    # Sampling parameters
    start = 3000  # Skip initial equilibration steps
    sample_step = 1
    
    plt.figure(figsize=(8, 6))
    
    # Store data for normalization analysis
    all_data = {}
    
    for i, Lx in enumerate(lattice_sizes):
        # Construct filename for this lattice size
        Ltau = int(40 * Lx)
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_True_step_10000.pt"
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
        
        # Average over equilibrated timesteps and batch dimension
        spsm_r_eq = spsm_r[seq_idx].mean(dim=0)  # Average over timesteps: [batch_size, Ly, Lx]
        spsm_r_avg = spsm_r_eq.mean(dim=0)       # Average over batches: [Ly, Lx]
        spsm_r_avg = spsm_r_avg.abs()            # Take absolute value for correlation
        
        # Convert to numpy for easier manipulation
        spsm_r_np = spsm_r_avg.numpy()
    
        r_values = []
        spin_corr_values = []
        spin_corr_errors = []
        
        # Simplified: plot spin correlation along x-direction only (y=0)
        for r in range(1, Lx // 2 + 1, 2):
            x = r
            y = 0  
            
            r_values.append(r)
            spin_corr_values.append(spsm_r_np[y, x])
            
            # Calculate error from all equilibrated samples at this position
            corr_samples_at_r = []
            for seq_step in seq_idx:
                # Average over batch for this timestep at position (x,y)
                corr_samples_at_r.append(spsm_r[seq_step, :, x, y].mean().item())
            
            spin_corr_errors.append(np.std(corr_samples_at_r) / np.sqrt(len(corr_samples_at_r)))
        
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
        r_fit = np.array(r_values[0:5])
        spin_corr_fit = np.array(spin_corr_values[0:5])
        spin_corr_err_fit = np.array(spin_corr_errors[0:5])

        # Linear fit in log-log space
        log_r = np.log(r_fit)
        log_corr = np.log(spin_corr_fit)
        coeffs = np.polyfit(log_r, log_corr, 1)
        fit_line = np.exp(coeffs[1]) * r_fit**coeffs[0]

        # Plot data and fit in log-log space
        plt.errorbar(r_values[0:], spin_corr_values[0:], yerr=spin_corr_errors[0:], 
                     linestyle='', marker='o', lw=1.5, color=color, 
                     label=f'{Lx}x{Ltau}', markersize=8, alpha=0.8)
        plt.plot(r_fit, fit_line, '-', color=color, alpha=0.6, lw=1.5, 
                 label=f'Fit L={Lx}: y~x^{coeffs[0]:.2f}')
        
        dbstop = 1
    
    # Plot the r_l20 and corr_l20 data on the same plot for comparison
    plt.plot(r_l20, corr_l20, 's--', color='black', label='L20 dqmc', markersize=8, alpha=0.8)
        
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Distance r (lattice units)', fontsize=14)
    plt.ylabel('Spin-Spin Correlation $\\langle S(0) S(r) \\rangle$', fontsize=14)

    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot (linear axes)
    save_dir = os.path.join(script_path, "./figures/spin_r_fit")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "spin_r_vs_x_fit.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Raw values figure saved at: {file_path}")


    plt.show()

 

if __name__ == '__main__':
    plot_spin_r()
    
    dbstop = 1


