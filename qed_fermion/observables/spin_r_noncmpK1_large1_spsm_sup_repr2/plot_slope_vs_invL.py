import re
import matplotlib.pyplot as plt

plt.ion()

import numpy as np
from scipy.optimize import curve_fit, minimize

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

start_dist = 1
step_dist = 2 
y_diplacement = lambda x: 0

# HMC data folder
data_folder = "/Users/kx/Desktop/hmc/fignote/back_tracing/hmc_check_point_noncmpK1_large1_spsm_sup_repr2"

# Set default plotting settings for physics scientific publication (Matlab style)
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()  

def linear_func(x, slope, intercept):
    """Linear function for fitting: y = slope * x + intercept"""
    return slope * x + intercept

def fit_slope_with_error(log_r, log_corr, log_corr_errors):
    """
    Perform unweighted linear fit in log-log space with error estimation.
    
    Parameters:
    -----------
    log_r : array
        Log of r values
    log_corr : array
        Log of correlation values
    log_corr_errors : array
        Errors in log space (not used for unweighted fit, kept for compatibility)
    
    Returns:
    --------
    slope : float
        Fitted slope
    slope_error : float
        Error in slope
    intercept : float
        Fitted intercept
    """
    # Use unweighted fit (numpy.polyfit)
    coeffs = np.polyfit(log_r, log_corr, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate error in slope using residual sum of squares
    y_pred = slope * log_r + intercept
    residuals = log_corr - y_pred
    ss_res = np.sum(residuals**2)
    n = len(log_r)
    mse = ss_res / (n - 2)  # Mean squared error (degrees of freedom = n - 2)
    
    # Calculate variance of slope
    ss_x = np.sum((log_r - np.mean(log_r))**2)
    slope_variance = mse / ss_x
    slope_error = np.sqrt(slope_variance)
    
    return slope, slope_error, intercept

def plot_slope_vs_invL():
    """Plot slope of log-log fit vs 1/L for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Create first figure for correlation plots with fits
    plt.figure(figsize=(8, 6))
    main_ax = plt.gca()
    
    # Finalize first figure: log-log correlation plots with fits
    main_ax.set_xlabel('r', fontsize=23)
    main_ax.set_ylabel(r'$C_S^{\uparrow\downarrow}(r, 0)$', fontsize=23)
    main_ax.set_xscale('log')
    main_ax.set_yscale('log')
    main_ax.grid(True, alpha=0.3)
    main_ax.xaxis.set_tick_params(labelsize=22)
    main_ax.yaxis.set_tick_params(labelsize=22)
    main_ax.legend(fontsize=14, ncol=2, loc='best', framealpha=0.9)
    
    plt.tight_layout()

    # Store results
    slopes = []
    slope_errors = []
    inv_L_values = []
    Lx_values = []
    
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

        r_values = np.array(r_values)
        spin_corr_values = np.array(spin_corr_values)
        spin_corr_errors = np.array(spin_corr_errors)
        
        # Determine fit window based on Lx
        n_points = len(r_values)
        if Lx <= 16:
            # Fit from 1st to 1/2*len(r_values)-th data (both inclusive)
            # 0-indexed: from 0 to n_points//2 - 1
            start_idx = 0
            end_idx = (n_points + 1) // 2 - 1
        else:
            # Fit from 3rd to min(1/2*len(r_values)-th, 8th) data (both inclusive)
            # 0-indexed: from 2 to min(n_points//2 - 1, 7)
            start_idx = 0
            end_idx = min((n_points + 1) // 2 - 1, 7)
        
        # Ensure valid indices
        if end_idx < start_idx or start_idx >= n_points:
            print(f"Warning: Invalid fit window for Lx={Lx}, skipping...")
            continue
        
        # Extract data for fitting
        r_fit = r_values[start_idx:end_idx+1]
        corr_fit = spin_corr_values[start_idx:end_idx+1]
        corr_err_fit = spin_corr_errors[start_idx:end_idx+1]
        
        # Convert to log space
        log_r = np.log(r_fit)
        log_corr = np.log(corr_fit)
        
        # Calculate errors in log space using error propagation
        # For log(y), error is approximately err_y / y (relative error)
        log_corr_errors = corr_err_fit / corr_fit
        
        # Perform linear fit with error estimation
        slope, slope_error, intercept = fit_slope_with_error(log_r, log_corr, log_corr_errors)
        
        # Generate fit line for visualization (extend slightly beyond fit points for better visualization)
        r_fit_extended = np.linspace(r_fit[0], r_fit[-1] * 1.0, 100)
        fit_line = np.exp(intercept) * r_fit_extended**slope
        
        print(f"Lx={Lx}: slope={slope:.4f} ± {slope_error:.4f}, fit window: indices {start_idx} to {end_idx}")
        
        # Store results
        slopes.append(slope)
        slope_errors.append(slope_error)
        inv_L_values.append(1.0 / Lx)
        Lx_values.append(Lx)
        
        # Plot data and fit during iteration
        color = f"C{i}"
        Ltau = int(10 * Lx)
        label = rf'{Ltau}x{Lx}$^2$'
        
        # Plot data
        main_ax.errorbar(r_values, spin_corr_values, yerr=spin_corr_errors,
                         linestyle=':', marker='o', color=color,
                         markersize=8, alpha=0.7, label=label)
        
        # Plot fit line
        main_ax.plot(r_fit_extended, fit_line, '-', color=color, alpha=0.8, lw=2)
    
        dbstop = 1
    
    # Save the first plot
    save_dir = os.path.join(script_path, "./figures/slope_vs_invL")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path1 = os.path.join(save_dir, "corr_with_fits_noncmpK1.pdf")
    png_path1 = os.path.join(save_dir, "corr_with_fits_noncmpK1.png")
    plt.savefig(pdf_path1, format="pdf", bbox_inches="tight")
    plt.savefig(png_path1, format="png", bbox_inches="tight", dpi=300)
    print(f"Correlation plot with fits saved at: {pdf_path1}")
    
    plt.show()
    
    # Second figure: slope vs 1/L
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    
    ax2.errorbar(inv_L_values, slopes, yerr=slope_errors, 
                marker='o', markersize=10, linestyle='-', linewidth=2,
                capsize=5, capthick=2, elinewidth=2, alpha=0.8, label='Data')
    
    # Fit with a function that flattens as 1/L -> 0 (nonincreasing)
    # Using a rational function: y = a + b * (1/L) / (1 + c * (1/L))
    # This naturally flattens to y = a as 1/L -> 0, with derivative at 0 = b
    # To ensure nonincreasing, we constrain b <= 0
    inv_L_array = np.array(inv_L_values)
    slopes_array = np.array(slopes)
    Lx_array = np.array(Lx_values)
    
    # Calculate weights: weight proportional to L^2 (larger systems get more weight)
    weights = Lx_array**2
    weights = weights / np.mean(weights)
    
    def power_rat_func(x, a, b, c, d):
        """
        Power-law rational function: y = a + b * x^d / (1 + c * x)
        - As x -> 0: y -> a (flat, derivative = 0 if d > 1)
        - As x increases: y increases
        - Derivative at 0: 0 if d > 1, or b if d = 1
        - For d > 1, derivative at 0 is 0 (nonincreasing/flat)
        """
        if len(x) == 0:
            return np.array([])
        x_safe = np.maximum(x, 1e-10)  # Avoid issues with x=0
        return a + b * (x_safe**d) / (1 + c * x_safe)
    
    # Initial guess
    a_init = np.min(slopes_array)  # Value at 1/L=0
    b_init = (np.max(slopes_array) - np.min(slopes_array)) * 10.0  # Amplitude
    c_init = 10.0  # Controls saturation
    d_init = 2.0  # Power (should be > 1 for flat derivative at 0)

    # Fit with weights
    popt, pcov = curve_fit(power_rat_func, inv_L_array, slopes_array, 
                            p0=[a_init, b_init, c_init, d_init],
                            sigma=1.0/np.sqrt(weights),
                            absolute_sigma=False,
                            bounds=([-np.inf, -np.inf, 0, 1.1], [np.inf, np.inf, np.inf, 5.0]))  # d > 1, c >= 0
    
    a_fit, b_fit, c_fit, d_fit = popt
    
    # Generate smooth curve for plotting
    inv_L_fit = np.linspace(0, max(inv_L_array) * 1.1, 200)
    slopes_fit = power_rat_func(inv_L_fit, a_fit, b_fit, c_fit, d_fit)
    
    # Extrapolate to 1/L = 0
    slope_extrapolated = power_rat_func(np.array([0.0]), a_fit, b_fit, c_fit, d_fit)[0]
    
    # Calculate R-squared
    y_pred = power_rat_func(inv_L_array, a_fit, b_fit, c_fit, d_fit)
    ss_res = np.sum(weights * (slopes_array - y_pred)**2)
    y_mean = np.average(slopes_array, weights=weights)
    ss_tot = np.sum(weights * (slopes_array - y_mean)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Check derivative at 0 (should be 0 if d > 1)
    # For y = a + b * x^d / (1 + c * x), derivative at 0 is 0 if d > 1
    deriv_at_0 = 0.0 if d_fit > 1.0 else b_fit
    
    print(f"Power-rational fit: a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}, d = {d_fit:.4f}")
    print(f"Weighted R² = {r2:.4f}, derivative at 0 = {deriv_at_0:.4f}")
    print(f"Extrapolated slope at 1/L = 0: {slope_extrapolated:.4f}")

    # Plot the fit line
    ax2.plot(inv_L_fit, slopes_fit, '--', color='red', linewidth=2, 
                label=f'Power-rational fit (R²={r2:.3f})')
    
    # Mark the extrapolated point at 1/L = 0
    ax2.plot([0], [slope_extrapolated], 's', color='red', markersize=12, 
                label=f'Extrapolated: {slope_extrapolated:.4f}', zorder=5)
    
    # Set x-axis to include 0 to show extrapolated point
    ax2.set_xlim(left=0)
    
    ax2.legend(fontsize=14, loc='best')
        
    ax2.set_xlabel(r'$1/L$', fontsize=23)
    ax2.set_ylabel('$2\Delta$', fontsize=23)
    ax2.grid(True, alpha=0.3)
    
    # Set tick label size
    ax2.xaxis.set_tick_params(labelsize=22)
    ax2.yaxis.set_tick_params(labelsize=22)
    
    plt.tight_layout()
    
    # Save the second plot
    pdf_path2 = os.path.join(save_dir, "slope_vs_invL_noncmpK1.pdf")
    png_path2 = os.path.join(save_dir, "slope_vs_invL_noncmpK1.png")
    plt.savefig(pdf_path2, format="pdf", bbox_inches="tight")
    plt.savefig(png_path2, format="png", bbox_inches="tight", dpi=300)
    print(f"Slope vs 1/L plot saved at: {pdf_path2}")
    
    plt.show()
    
    # Print summary
    print("\nSummary:")
    print("Lx\t1/L\t\tSlope\t\tSlope Error")
    print("-" * 50)
    for Lx, inv_L, slope, slope_err in zip(Lx_values, inv_L_values, slopes, slope_errors):
        print(f"{Lx}\t{inv_L:.6f}\t{slope:.4f}\t{slope_err:.4f}")

 
if __name__ == '__main__':
    plot_slope_vs_invL()
    
    dbstop = 1
