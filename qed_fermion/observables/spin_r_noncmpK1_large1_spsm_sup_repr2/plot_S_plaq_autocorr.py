import re
import matplotlib.pyplot as plt

plt.ion()

import numpy as np
from scipy.optimize import curve_fit

from matplotlib import rcParams
rcParams['figure.raise_window'] = False
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

EQUILIBRIUM_THRESHOLD = 0.065  # S_plaq density threshold for equilibrium

def exponential_decay(k, A, tau, offset):
    """Exponential decay function: A * exp(-k/tau) + offset"""
    return A * np.exp(-k / tau) + offset

def compute_autocorrelation(data, max_lag=None):
    """
    Compute autocorrelation function for a time series.
    
    Parameters:
    -----------
    data : array
        Time series data (1D array)
    max_lag : int, optional
        Maximum lag to compute. If None, uses len(data) // 2
    
    Returns:
    --------
    lags : array
        Lag values (k)
    autocorr : array
        Autocorrelation values at each lag
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 2
    
    # Normalize data (subtract mean)
    data_mean = np.mean(data)
    data_centered = data - data_mean
    
    # Compute variance
    variance = np.var(data_centered)
    if variance < 1e-10:
        return np.array([]), np.array([])
    
    # Compute autocorrelation for each lag
    lags = np.arange(0, min(max_lag, n))
    autocorr = np.zeros(len(lags))
    
    for i, k in enumerate(lags):
        if k == 0:
            autocorr[i] = 1.0
        else:
            # Autocorrelation at lag k: E[(X_t - mu)(X_{t+k} - mu)] / var(X)
            numerator = np.mean(data_centered[:-k] * data_centered[k:])
            autocorr[i] = numerator / variance
    
    return lags, autocorr

def fit_autocorr_length(lags, autocorr):
    """
    Fit exponential decay to autocorrelation function to extract autocorrelation length.
    
    Parameters:
    -----------
    lags : array
        Lag values (k)
    autocorr : array
        Autocorrelation values
    
    Returns:
    --------
    tau : float
        Autocorrelation length (NaN if fit fails)
    tau_err : float
        Error in tau (NaN if fit fails)
    fit_params : tuple or None
        Fitted parameters (A, tau, offset) if successful, None otherwise
    """
    if len(lags) < 10 or len(autocorr) < 10:
        return np.nan, np.nan, None
    
    # Fit only the initial consecutive positive sequence of autocorrelation values starting from lag 1
    mask = lags > 0
    lags_pos = lags[mask]
    autocorr_pos = autocorr[mask]
    # Find the longest initial sequence of positive autocorr (starting at k=1)
    pos_idx = np.where(autocorr_pos > 0)[0]
    if len(pos_idx) == 0 or pos_idx[0] != 0:
        return np.nan, np.nan, None

    # Find where initial positive sequence ends
    first_nonpos = np.where(autocorr_pos <= 0)[0]
    if len(first_nonpos) == 0:
        end = len(autocorr_pos)
    else:
        end = first_nonpos[0]
    k_fit = lags_pos[:end]
    autocorr_fit = autocorr_pos[:end]
    if len(k_fit) < 5:
        return np.nan, np.nan, None
    
    # Initial guess: A = max autocorr (should be ~1), tau = some fraction of max lag
    A_guess = np.max(autocorr_fit)
    tau_guess = k_fit[-1] / 3.0  # Rough guess
    offset_guess = np.min(autocorr_fit)  # Should be close to 0
    
    try:
        # Fit exponential decay
        popt, pcov = curve_fit(exponential_decay, k_fit, autocorr_fit,
                              p0=[A_guess, tau_guess, offset_guess],
                              bounds=([0, 1, -0.1], [2.0, len(k_fit), 0.1]),
                              maxfev=5000)
        tau = popt[1]
        tau_err = np.sqrt(pcov[1, 1]) if not np.isnan(pcov[1, 1]) else np.nan
        fit_params = (popt[0], popt[1], popt[2])  # (A, tau, offset)
        return tau, tau_err, fit_params
    except:
        return np.nan, np.nan, None

def plot_S_plaq_timestep():
    """Plot autocorrelation-k curves for S_plaq density for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Create figures
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.33))
    
    # Store autocorrelation lengths
    corr_lengths = {}
    corr_lengths_err = {}
    
    for i, Lx in enumerate(lattice_sizes):
        Ltau = int(10 * Lx)
        start = 2000 if Lx >= 20 else 3000
        # start = 0
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

        # Extract S_plaq data: S_plaq_list
        S_plaq = res['S_plaq_list']  # Shape: [timesteps, batch_size]

        # Extract sequence indices for equilibrated samples
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))
        seq_idx = np.arange(start, end, sample_step)

        # Average over batch dimension (axis=1) and convert to numpy
        # S_plaq shape: [timesteps, batch_size]
        S_plaq_avg = S_plaq.mean(axis=1).cpu().numpy()
        
        # Compute density: normalize by space-time volume (Lx * Ly * Ltau)
        # Since Ly = Lx, volume = Lx^2 * Ltau
        volume = Lx * Lx * Ltau  # Lx * Ly * Ltau
        S_plaq_density = np.abs(S_plaq_avg) / volume
        
        # Find equilibrium point: first time S_plaq_density >= EQUILIBRIUM_THRESHOLD
        # Set EQUILIBRIUM_THRESHOLD to mean of S_plaq_density[5000:6000]
        EQUILIBRIUM_THRESHOLD = (S_plaq_density[5000:6000].mean() if Lx >= 30 else S_plaq_density[3000:4000].mean()) * 1.1
        equilibrium_indices = np.where(S_plaq_density <= EQUILIBRIUM_THRESHOLD)[0]
        if len(equilibrium_indices) == 0:
            print(f'Lx={Lx}: No equilibrium point found (S_plaq_density never reached {EQUILIBRIUM_THRESHOLD})')
            continue
        
        equilibrium_start_idx = equilibrium_indices[0]
        equilibrium_data = S_plaq_density[equilibrium_start_idx:]
        
        print(f'Lx={Lx}: Equilibrium {EQUILIBRIUM_THRESHOLD:.3g} starts at step {equilibrium_start_idx}, '
              f'using {len(equilibrium_data)} equilibrium points')
        
        # Compute autocorrelation function
        lags, autocorr = compute_autocorrelation(equilibrium_data)
        mask = lags <= 3000
        lags = lags[mask]
        autocorr = autocorr[mask]
        
        
        if len(lags) == 0 or len(autocorr) == 0:
            print(f'Lx={Lx}: Failed to compute autocorrelation')
            continue
        
        # Fit exponential decay to autocorrelation to extract tau_L
        tau, tau_err, fit_params = fit_autocorr_length(lags, autocorr)
        
        if not np.isnan(tau) and Lx <= 40:
            corr_lengths[Lx] = tau
            corr_lengths_err[Lx] = tau_err
            print(f'Lx={Lx}: tau_L={tau:.2f} ± {tau_err:.2f}')
        
        # Plot autocorrelation-k curve
        # Subsample for plotting if too many points
        max_plot_points = 200
        if len(lags) > max_plot_points:
            subsample_step = len(lags) // max_plot_points
            plot_indices = np.arange(0, len(lags), subsample_step)
            lags_plot = lags[plot_indices]
            autocorr_plot = autocorr[plot_indices]
        else:
            lags_plot = lags
            autocorr_plot = autocorr
        
        ax.plot(lags_plot, autocorr_plot, 'o-', label=f'$L={Lx}$', 
               alpha=0.7, markersize=3, linewidth=1.5)

        # Create a supplementary figure for the autocorrelation curves (for publication or SI)
        # supp_fig, supp_ax = plt.subplots(figsize=(6, 4.2))
        # supp_ax.plot(lags_plot, autocorr_plot, 'o-', label=f'$L={Lx}$', 
        #              alpha=0.7, markersize=3, linewidth=1.5)
        # supp_ax.set_xlabel("Lag $k$", fontsize=14)
        # supp_ax.set_ylabel("Autocorrelation", fontsize=14)
        # supp_ax.grid(True, alpha=0.3, which='both')

        # Plot fit if successful
        if not np.isnan(tau) and fit_params is not None:
            # Generate smooth fit curve for positive lags where autocorr > 0
            mask = (0 < lags) & (autocorr > 0)
            if np.sum(mask) > 0:
                k_min = lags[mask].min()
                k_max = lags[mask].max()
                # Generate denser k values for smooth curve
                k_fit_smooth = np.linspace(k_min, k_max, 200)
                A_fit, tau_fit, offset_fit = fit_params
                autocorr_fit_values = exponential_decay(k_fit_smooth, A_fit, tau_fit, offset_fit)
                # Only plot where fit values are positive (for log scale)
                mask_positive = autocorr_fit_values > 0
                if np.sum(mask_positive) > 0:
                    ax.plot(k_fit_smooth[mask_positive], autocorr_fit_values[mask_positive], '--', 
                           alpha=0.8, linewidth=1.5, color=ax.lines[-1].get_color())
        else:
            dbstop = 1
        
        dbstop = 1
        # ax.legend(fontsize=11, ncol=3, loc='lower left')

    
    ax.set_xlabel("Lag $k$", fontsize=14)
    ax.set_ylabel("Autocorrelation", fontsize=14)
    ax.set_xlim(left=0, right=3000)
    ax.set_ylim(bottom=-0.5, top=1)
    # ax.set_yscale('log')
    ax.legend(fontsize=11, ncol=4, loc='lower left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add inset plot for autocorrelation length vs lattice size
    if len(corr_lengths) > 0:
        Lx_sorted = sorted(corr_lengths.keys())
        tau_values = [corr_lengths[Lx] for Lx in Lx_sorted]
        tau_errors = [corr_lengths_err.get(Lx, 0) for Lx in Lx_sorted]
        
        inset_width = 0.58
        inset_height = 0.62
        inset_ax = inset_axes(
            ax,
            width=f"{inset_width*100}%", height=f"{inset_height*100}%",
            bbox_to_anchor=(0.35, 0.35, inset_width, inset_height),
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        
        # Plot autocorrelation length vs lattice size in inset and fit power-law
        inset_ax.errorbar(Lx_sorted, tau_values, yerr=tau_errors, 
                         fmt='k^', linewidth=2, markersize=6, capsize=4)
        inset_ax.set_xlabel("$L$", fontsize=13)
        inset_ax.set_ylabel("$\\tau_L$", fontsize=13)
        inset_ax.grid(True, alpha=0.3)
        inset_ax.tick_params(axis='both', which='major', labelsize=13)
        inset_ax.xaxis.set_tick_params(labelsize=13)
        inset_ax.yaxis.set_tick_params(labelsize=13)
        
        # Fit to a power-law: tau_L = a * L**z
        from scipy.optimize import curve_fit

        def power_law(L, a, z):
            return a * L**z

        # Only use points with finite values for fitting
        Lx_arr = np.array(Lx_sorted)
        tau_arr = np.array(tau_values)
        tau_err_arr = np.array(tau_errors)
        valid = np.isfinite(Lx_arr) & np.isfinite(tau_arr) & (tau_arr > 0)
        if np.sum(valid) >= 2:
            try:
                popt, pcov = curve_fit(
                    power_law,
                    Lx_arr[valid],
                    tau_arr[valid],
                    p0=[1.0, 1.0],
                    sigma=tau_err_arr[valid] if len(tau_err_arr) == len(Lx_arr) and np.all(tau_err_arr[valid] > 0) else None,
                    absolute_sigma=True if len(tau_err_arr) == len(Lx_arr) and np.all(tau_err_arr[valid] > 0) else False
                )
                a_fit, z_fit = popt
                err_a, err_z = np.sqrt(np.diag(pcov))
                # Plot the fit line
                Lx_fit = np.linspace(min(Lx_arr[valid]), max(Lx_arr[valid]), 200)
                tau_fit = power_law(Lx_fit, a_fit, z_fit)
                inset_ax.plot(Lx_fit, tau_fit, 'b--', lw=2, label=fr"$\sim L^{{{z_fit:.2f}}}$")
                # Annotate the exponent
                inset_ax.text(0.05, 0.9, fr"$z = {z_fit:.2f} \pm {err_z:.2f}$", transform=inset_ax.transAxes, 
                              fontsize=12, verticalalignment='top', color='b')
                inset_ax.legend(fontsize=11, frameon=False)
            except Exception as e:
                print("Power-law fit failed:", e)

    
    # Add "(b)" label at top left corner, aligned with y-axis label
    ax.text(-0.13, 0.98, "(b)", transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(script_path, "./figures/S_plaq_autocorr")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "S_plaq_density_autocorr_noncmpK1.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show()


if __name__ == '__main__':
    plot_S_plaq_timestep()
