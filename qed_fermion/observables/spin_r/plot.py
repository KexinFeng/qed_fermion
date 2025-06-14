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



def plot_spin_r():
    """Plot spin-spin correlation as a function of distance r for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40]
    lattice_sizes = [10, 12]
    
    # HMC data folder
    hmc_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/cmp_large/hmc_check_point_large"
    
    # Sampling parameters
    start = 2000  # Skip initial equilibration steps
    sample_step = 1
    
    plt.figure(figsize=(10, 8))
    
    for i, Lx in enumerate(lattice_sizes):
        # Construct filename for this lattice size
        Ltau = int(10 * Lx)
        hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_True_step_10000.pt"
        hmc_filename = os.path.join(hmc_folder, hmc_file)
        
        if not os.path.exists(hmc_filename):
            print(f"File not found: {hmc_filename}")
            continue
            
        # Load checkpoint data
        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')
        
        # Extract spin-spin correlation data: spsm_r_list
        spsm_r = res['spsm_r_list']  # Shape: [timesteps, batch_size, Lx, Ly]
        
        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)
        
        # Average over equilibrated timesteps and batch dimension
        spsm_r_eq = spsm_r[seq_idx].mean(dim=0)  # Average over timesteps: [batch_size, Lx, Ly]
        spsm_r_avg = spsm_r_eq.mean(dim=0)       # Average over batches: [Lx, Ly]
        
        # Convert to numpy for easier manipulation
        spsm_r_np = spsm_r_avg.numpy()
        
        # Calculate radial average: spin correlation as function of distance r
        center_x, center_y = Lx // 2, Lx // 2  # Assuming square lattice
        max_r = min(center_x, center_y)
        
        r_values = []
        spin_corr_values = []
        spin_corr_errors = []
        
        for r in range(1, max_r):
            # Find all points at distance r from center
            corr_at_r = []
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if int(np.sqrt(dx*dx + dy*dy)) == r:
                        x = (center_x + dx) % Lx
                        y = (center_y + dy) % Lx
                        corr_at_r.append(spsm_r_np[x, y])
            
            if corr_at_r:
                r_values.append(r)
                spin_corr_values.append(np.mean(corr_at_r))
                
                # Calculate error from all equilibrated samples at this r
                corr_samples_at_r = []
                for seq_step in seq_idx:
                    corr_step = []
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            if int(np.sqrt(dx*dx + dy*dy)) == r:
                                x = (center_x + dx) % Lx
                                y = (center_y + dy) % Lx
                                # Average over batch for this timestep
                                corr_step.append(spsm_r[seq_step, :, x, y].mean().item())
                    if corr_step:
                        corr_samples_at_r.append(np.mean(corr_step))
                
                spin_corr_errors.append(np.std(corr_samples_at_r) / np.sqrt(len(corr_samples_at_r)))
        
        # Plot spin correlation vs distance for this lattice size
        color = f"C{i}"
        plt.errorbar(r_values, spin_corr_values, yerr=spin_corr_errors, 
                    linestyle='-', marker='o', lw=2, color=color, 
                    label=f'L={Lx}', markersize=4, alpha=0.8)
    
    plt.xlabel('Distance r', fontsize=14)
    plt.ylabel('Spin-Spin Correlation $\\langle S(0) S(r) \\rangle$', fontsize=14)
    plt.title('Spin-Spin Correlation vs Distance for Different Lattice Sizes', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often reveals exponential decay
    plt.tight_layout()
    
    # Save the plot
    save_dir = os.path.join(script_path, "./figures/spin_r_comparison")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "spin_r_vs_distance_all_sizes.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")
    
    plt.show()
    
    return


if __name__ == '__main__':
    plot_spin_r()
    
    dbstop = 1


