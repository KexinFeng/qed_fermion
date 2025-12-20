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

from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()

# HMC data folder
data_folder = "/Users/kx/Desktop/hmc/fignote/back_tracing/hmc_check_point_noncmpK1_large1_spsm_sup_repr2"

# Set default plotting settings for physics scientific publication (Matlab style)
set_default_plotting()

def plot_S_tau_timestep():
    """Plot S_tau versus time step for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Create a single figure for all plots
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, Lx in enumerate(lattice_sizes):
        Ltau = int(10 * Lx)
        start = 2000 if Lx >= 20 else 4000
        start = 0
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

        # Load data
        hmc_filename = find_hmc_file(Lx, Ltau)
        if hmc_filename is None:
            continue
        
        # Load checkpoint data
        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')

        # Extract S_tau data: S_tau_list
        S_tau = res['S_tau_list']  # Shape: [timesteps, batch_size]

        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)
        seq_idx_all = np.arange(end)

        # Average over batch dimension (axis=1) and convert to numpy
        # S_tau shape: [timesteps, batch_size]
        S_tau_avg = S_tau[seq_idx].mean(axis=1).cpu().numpy()
        
        # Plot S_tau vs time step (similar to total_monitoring which uses '*' marker)
        ax.plot(seq_idx, S_tau_avg, '*', label=f'{Ltau}x{Lx}$^2$', alpha=0.7, markersize=6)
    
    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("$S_{tau}$", fontsize=14)
    ax.set_title("$S_{tau}$ Over Steps", fontsize=16)
    ax.legend(fontsize=10, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(script_path, "./figures/S_tau_timestep")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "S_tau_vs_timestep_noncmpK1.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show()


if __name__ == '__main__':
    plot_S_tau_timestep()
