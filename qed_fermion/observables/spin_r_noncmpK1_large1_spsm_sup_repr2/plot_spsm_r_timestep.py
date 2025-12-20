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

def plot_spsm_r_timestep():
    """Plot spsm_r versus time step for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Create a figure with subplots for each lattice size
    # Arrange in a grid: calculate number of rows and columns
    n_lattices = len(lattice_sizes)
    n_cols = 3
    n_rows = (n_lattices + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
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

        # Extract spin-spin correlation data: spsm_r_list
        spsm_r = res['spsm_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]

        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)
        seq_idx_all = np.arange(end)

        # Plot spsm_r vs time step for specific r values (similar to total_monitoring)
        ax = axes[i]
        
        # Plot specific r values (similar to total_monitoring which plots r=3 and r=5)
        # Use r values that exist in the lattice
        r_values_to_plot = [3, 5]
        # Adjust r values if they're too large for the lattice
        r_values_to_plot = [r for r in r_values_to_plot if r < Lx]
        
        for r in r_values_to_plot:
            # Average over batch dimension (axis=1) and take absolute value
            # spsm_r shape: [timesteps, batch_size, Ly, Lx]
            # We want to plot at y=0, x=r
            spsm_r_at_r = spsm_r[seq_idx, :, 0, r].abs().mean(axis=1).numpy()
            ax.plot(seq_idx, spsm_r_at_r, label=f'spsm_r[{r}]')
        
        ax.set_xlabel("Steps", fontsize=14)
        ax.set_ylabel("Spsm_r", fontsize=14)
        ax.set_title(f"spsm_r Over Steps - {Ltau}x{Lx}$^2$", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_lattices, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(script_path, "./figures/spsm_r_timestep")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "spsm_r_vs_timestep_noncmpK1.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show()


if __name__ == '__main__':
    plot_spsm_r_timestep()
