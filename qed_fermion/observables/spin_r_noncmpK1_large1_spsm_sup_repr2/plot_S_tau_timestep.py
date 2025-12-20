import re
import matplotlib.pyplot as plt

plt.ion()

import numpy as np
from scipy.optimize import curve_fit

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

def exponential_decay(t, A, tau, offset):
    """Exponential decay function: A * exp(-t/tau) + offset"""
    return A * (1 - np.exp(-t / tau)) + offset

def fit_autocorr_length(seq_idx, density, thermalization_skip=500, tail_fraction=0.2):
    """
    Fit exponential decay to extract autocorrelation length.
    
    Parameters:
    -----------
    seq_idx : array
        Time step indices
    density : array
        Density values
    thermalization_skip : int
        Number of initial steps to skip (thermalization)
    tail_fraction : float
        Fraction of data from the end to use for estimating equilibrium value
    
    Returns:
    --------
    tau : float
        Autocorrelation length (NaN if fit fails)
    tau_err : float
        Error in tau (NaN if fit fails)
    density_eq : float
        Estimated equilibrium density
    fit_params : tuple or None
        (A, tau, offset) fitted parameters, or None if fit failed
    """
    # Estimate equilibrium value from the tail
    tail_size = int(len(density) * tail_fraction)
    density_eq = np.mean(density[-tail_size:])
    
    # Skip thermalization period
    skip_idx = min(thermalization_skip, len(seq_idx) // 4)
    fit_start_idx = skip_idx
    fit_end_idx = len(seq_idx)
    
    # Extract data for fitting
    t_fit = seq_idx[fit_start_idx:fit_end_idx] - seq_idx[fit_start_idx]  # Start from 0
    density_fit = density[fit_start_idx:fit_end_idx]
    
    # Only fit if we have enough points
    if len(t_fit) < 10:
        return np.nan, np.nan, density_eq, None
    
    # For relaxation form A * (1 - exp(-t/tau)) + offset:
    # - At t=0: offset (initial value)
    # - As t→∞: A + offset (final/equilibrium value)
    # So: offset = initial value, A = (final - initial)
    initial_value = np.mean(density_fit[:min(50, len(density_fit)//10)])  # Average of first few points
    final_value = np.mean(density_fit[-min(50, len(density_fit)//10):])   # Average of last few points
    
    # Initial guess for parameters
    offset_guess = initial_value  # Starting value
    A_guess = final_value - initial_value  # Change from initial to final
    # tau guess: time for ~63% of the change (1 - exp(-1) ≈ 0.63)
    # Estimate where we're at 63% of the way
    if abs(A_guess) > 1e-10:
        target_value = initial_value + 0.63 * A_guess
        # Find index where density is closest to target
        idx_63 = np.argmin(np.abs(density_fit - target_value))
        if idx_63 > 0 and idx_63 < len(t_fit):
            tau_guess = t_fit[idx_63]
        else:
            tau_guess = (t_fit[-1] - t_fit[0]) / 3.0  # Fallback: 1/3 of total time
    else:
        tau_guess = (t_fit[-1] - t_fit[0]) / 3.0
    
    # Ensure positive values for the fit
    if A_guess < 0:
        # If decreasing, we might need to adjust, but for now keep it
        A_guess = abs(A_guess)
    
    try:
        # Fit relaxation function: A * (1 - exp(-t/tau)) + offset
        popt, pcov = curve_fit(exponential_decay, t_fit, density_fit,
                              p0=[A_guess, tau_guess, offset_guess],
                              bounds=([0, 1, density_fit.min()*0.5], 
                                     [abs(A_guess)*3, len(t_fit)*2, density_fit.max()*1.5]),
                              maxfev=5000)
        tau = popt[1]
        tau_err = np.sqrt(pcov[1, 1]) if not np.isnan(pcov[1, 1]) else np.nan
        fit_params = (popt[0], popt[1], popt[2])  # (A, tau, offset)
        return tau, tau_err, density_eq, fit_params
    except Exception as e:
        print(f"Fit failed: {e}")
        return np.nan, np.nan, density_eq, None

def plot_S_tau_timestep():
    """Plot S_tau versus time step for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Create figures
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.33))
    
    # Store autocorrelation lengths
    corr_lengths = {}
    lattice_list = []
        
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
        # Truncate to max 6500 steps
        end = min(end, start + 6500)
        seq_idx = np.arange(start, end, sample_step)
        seq_idx_all = np.arange(end)

        # Average over batch dimension (axis=1) and convert to numpy
        # S_tau shape: [timesteps, batch_size]
        S_tau_avg = S_tau[seq_idx].mean(axis=1).cpu().numpy()
        
        # Compute density: normalize by volume (Lx * Ly * Ltau)
        # Since Ly = Lx and Ltau = 10 * Lx, volume = Lx^2 * Ltau = Lx^2 * 10 * Lx = 10 * Lx^3
        volume = Lx * Lx * Ltau  # Lx * Ly * Ltau
        S_tau_density = np.abs(S_tau_avg) / volume
        
        # Fit autocorrelation length
        thermalization_skip = max(500, int(len(seq_idx) * 0.1))  # Skip at least 10% or 500 steps
        tau, tau_err, density_eq, fit_params = fit_autocorr_length(seq_idx, S_tau_density, 
                                                       thermalization_skip=thermalization_skip)
        
        if not np.isnan(tau):
            corr_lengths[Lx] = tau
            lattice_list.append(Lx)
            print(f'Lx={Lx}: tau={tau:.2f} ± {tau_err:.2f}, density_eq={density_eq:.6e}')
        
        # Subsample data points for sparser plotting (every Nth point)
        subsample_step = max(1, len(seq_idx) // 50)  # Aim for ~50 points max (much more sparse)
        plot_indices = np.arange(0, len(seq_idx), subsample_step)
        seq_idx_plot = seq_idx[plot_indices]
        S_tau_density_plot = S_tau_density[plot_indices]
        
        # Plot S_tau density vs time step (sparse points for better visibility)
        ax.plot(seq_idx_plot, S_tau_density_plot, '*', label=f'{Ltau}x{Lx}$^2$', 
               alpha=1.0, markersize=6)
        
        # Plot fit if successful (using fitted parameters)
        if not np.isnan(tau) and fit_params is not None:
            if Lx == 10: continue
            skip_idx = min(thermalization_skip, len(seq_idx) // 4)
            t_fit = seq_idx[skip_idx:] - seq_idx[skip_idx]
            # Use fitted parameters: A, tau, offset
            A_fit, tau_fit, offset_fit = fit_params
            density_fit = exponential_decay(t_fit, A_fit, tau_fit, offset_fit)
            ax.plot(seq_idx[skip_idx:], density_fit, '-', 
                   alpha=0.9, linewidth=1, color=ax.lines[-1].get_color())
    
    ax.set_ylim(0.86, 0.97)
    ax.set_xlim(left=-230, right=6700)
    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("$S_{\\tau}$ density", fontsize=14)
    # ax.set_title("$S_{tau}$ Density Over Steps", fontsize=16)
    ax.legend(fontsize=10, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(script_path, "./figures/S_tau_timestep")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "S_tau_density_vs_timestep_noncmpK1.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")
    
    # Plot autocorrelation length vs lattice size
    if len(corr_lengths) > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5.33))
        Lx_sorted = sorted(corr_lengths.keys())
        tau_values = [corr_lengths[Lx] for Lx in Lx_sorted]
        
        ax2.plot(Lx_sorted, tau_values, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel("$L$", fontsize=14)
        ax2.set_ylabel("$\\tau$", fontsize=14)
        # ax2.set_title("$S_{tau}$ Autocorrelation Length vs Lattice Size", fontsize=16)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_path2 = os.path.join(save_dir, "S_tau_corr_length_vs_size_noncmpK1.pdf")
        plt.savefig(file_path2, format="pdf", bbox_inches="tight")
        print(f"Correlation length figure saved at: {file_path2}")

    plt.show()


if __name__ == '__main__':
    plot_S_tau_timestep()
