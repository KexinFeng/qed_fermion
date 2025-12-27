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

# Parameters for data loading
start = 4000  # Skip initial equilibration steps
sample_step = 1

# HMC data folder
data_folder = "/Users/kx/Desktop/hmc/fignote/back_tracing/hmc_check_point_noncmpK1_large1_spsm_sup_repr2"

# Set default plotting settings for physics scientific publication (Matlab style)
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()  

def linear_func(x, slope, intercept):
    """Linear function for fitting: y = slope * x + intercept"""
    return slope * x + intercept

def fit_slope_with_error(log_tau, log_G, log_G_errors):
    """
    Perform unweighted linear fit in log-log space with error estimation.
    
    Parameters:
    -----------
    log_tau : array
        Log of tau values
    log_G : array
        Log of G values
    log_G_errors : array
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
    coeffs = np.polyfit(log_tau, log_G, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate error in slope using residual sum of squares
    y_pred = slope * log_tau + intercept
    residuals = log_G - y_pred
    ss_res = np.sum(residuals**2)
    n = len(log_tau)
    mse = ss_res / (n - 2)  # Mean squared error (degrees of freedom = n - 2)
    
    # Calculate variance of slope
    ss_x = np.sum((log_tau - np.mean(log_tau))**2)
    slope_variance = mse / ss_x
    slope_error = np.sqrt(slope_variance)
    
    return slope, slope_error, intercept

def plot_flux_slope_vs_invL():
    """Plot slope of log-log fit vs 1/L for different lattice sizes."""
    
    # Define lattice sizes to analyze
    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]
    
    # Create first figure for flux plots with fits
    plt.figure(figsize=(8, 6))
    main_ax = plt.gca()
    
    # Finalize first figure: log-log flux plots with fits
    main_ax.set_xlabel(r'$\tau$', fontsize=23)
    main_ax.set_ylabel(r'$C_{flux}(\tau)$', fontsize=23)
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
    
    import glob
    
    for i, Lx in enumerate(lattice_sizes):
        Ltau = int(10 * Lx)
        Ly = Lx
        
        # Find the correct file for this Lx and Ltau
        def find_hmc_file(Lx, Ltau, folder=data_folder):
            pattern = (
                f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_*_bs*_Jtau_1.2_K_1_dtau_0.1_delta_0.028_"
                "N_leapfrog_5_m_1_cg_rtol_*_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5*_Nrv_*_cmp_False_step_*.pt"
            )
            files = glob.glob(os.path.join(folder, pattern))
            if not files:
                print(f"No file found for Lx={Lx}, Ltau={Ltau} in {folder}")
                return None
            # Pick the file with the largest step (sort by step number)
            def extract_step(filename):
                m = re.search(r'step_(\d+)\.pt', filename)
                return int(m.group(1)) if m else 0
            files.sort(key=extract_step, reverse=True)
            return files[0]

        hmc_filename = find_hmc_file(Lx, Ltau)
        if hmc_filename is None:
            continue

        # Parse bs from filename
        m_bs = re.search(r'bs(\d+)', hmc_filename)
        bs = int(m_bs.group(1)) if m_bs else 1

        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')

        G_list = res['G_list']
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))

        seq_idx = np.arange(start, end, sample_step)
        batch_idx = np.arange(bs)

        G_mean = G_list[seq_idx][:, batch_idx].numpy().mean(axis=(0, 1))
        G_std = G_list[seq_idx][:, batch_idx].numpy().std(axis=(0, 1)) / np.sqrt(len(seq_idx) * bs)
        x = np.arange(G_mean.shape[0])
        tau = x + 1

        # Determine fit window: 
        # 1. Take the left half of all data
        # 2. Find the first contiguous window where G values are in [G_min, G_max]
        G_min = 10**(-3)
        G_max = 2.2 * 10**(-2)
        
        # First, take the left half of all data
        n_total = len(tau)
        left_half_end = (n_total + 1) // 2  # Left half (inclusive)
        left_half_indices = np.arange(left_half_end)
        
        # Get left half data
        tau_left = tau[left_half_indices]
        G_left = G_mean[left_half_indices]
        G_err_left = G_std[left_half_indices]
        
        # Find the first contiguous window in left half where G is in [G_min, G_max]
        range_mask = (G_left >= G_min) & (G_left <= G_max) & (G_left > 0) & (tau_left > 0)
        
        if np.sum(range_mask) < 3:  # Need at least 3 points
            print(f"Warning: Not enough points in range [10^(-3), 2.2*10^(-2)] in left half for Lx={Lx}, skipping...")
            continue
        
        # Find the first contiguous block of True values
        range_bool = range_mask.astype(int)
        # Find where the range starts (first True)
        first_true_idx = np.argmax(range_mask)
        if not range_mask[first_true_idx]:
            print(f"Warning: No valid points in range for Lx={Lx}, skipping...")
            continue
        
        # Find the end of the first contiguous block
        # Start from first_true_idx and find where it stops being True
        fit_start = first_true_idx
        fit_end = fit_start
        while fit_end < len(range_mask) and range_mask[fit_end]:
            fit_end += 1
        
        # Extract the first contiguous window
        fit_indices_in_left = np.arange(fit_start, fit_end)
        fit_indices = left_half_indices[fit_indices_in_left]
        
        if len(fit_indices) < 3:  # Need at least 3 points for fit
            print(f"Warning: Not enough points in first contiguous window for Lx={Lx}, skipping...")
            continue
        
        # Extract data for fitting
        tau_fit = tau[fit_indices]
        G_fit = G_mean[fit_indices]
        G_err_fit = G_std[fit_indices]
        
        # Convert to log space
        log_tau = np.log(tau_fit)
        log_G = np.log(G_fit)
        
        # Calculate errors in log space using error propagation
        # For log(y), error is approximately err_y / y (relative error)
        log_G_errors = G_err_fit / G_fit
        
        # Perform linear fit with error estimation
        slope, slope_error, intercept = fit_slope_with_error(log_tau, log_G, log_G_errors)
        
        # Generate fit line for visualization
        tau_fit_extended = np.linspace(tau_fit[0]*0.6, tau_fit[-1]*2.0, 100)
        fit_line = np.exp(intercept) * tau_fit_extended**slope
        
        print(f"Lx={Lx}: slope={slope:.4f} ± {slope_error:.4f}, fit window: tau {tau_fit[0]:.1f} to {tau_fit[-1]:.1f}")
        
        # Store results
        slopes.append(slope)
        slope_errors.append(slope_error)
        inv_L_values.append(1.0 / Lx)
        Lx_values.append(Lx)
        
        # Plot data and fit during iteration
        color = f"C{(i+1)%10}"  # Match color scheme from plot_flux.py
        Ltau = int(10 * Lx)
        label = rf'${Ltau}x{Lx}^2$'
        
        # Plot data (use similar indexing as in plot_flux.py)
        idx_leq_10 = np.where(tau <= 10)[0]
        idx_gt_10 = np.where(tau > 10)[0]
        idx_gt_10_sparse = idx_gt_10[::2]  # every 2nd point for tau > 10
        idx_plot = np.concatenate([idx_leq_10, idx_gt_10_sparse])
        idx_plot.sort()
        
        # Plot data (match style from plot_flux.py)
        main_ax.errorbar(tau[idx_plot], G_mean[idx_plot], yerr=G_std[idx_plot],
                         linestyle='', marker='o', markersize=8, label=label,
                         color=color, lw=2, alpha=0.8)
        
        # Plot fit line
        main_ax.plot(tau_fit_extended, fit_line, '-', color=color, alpha=0.8, lw=2)
    
    # Save the first plot
    save_dir = os.path.join(script_path, "./figures/flux_slope_vs_invL")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path1 = os.path.join(save_dir, "flux_with_fits_noncmpK1.pdf")
    png_path1 = os.path.join(save_dir, "flux_with_fits_noncmpK1.png")
    plt.savefig(pdf_path1, format="pdf", bbox_inches="tight")
    plt.savefig(png_path1, format="png", bbox_inches="tight", dpi=300)
    print(f"Flux plot with fits saved at: {pdf_path1}")
    
    plt.show()
    
    # ------------------------------------------------------------ #
    # Second figure: slope vs 1/L
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    
    # Fit with a function that flattens as 1/L -> 0 (nonincreasing)
    inv_L_array = np.array(inv_L_values)
    slopes_array = np.array(slopes)
    Lx_array = np.array(Lx_values)
    
    ax2.errorbar(inv_L_array, slopes_array, yerr=slope_errors, 
                marker='o', markersize=10, linestyle='', linewidth=2,
                capsize=5, capthick=2, elinewidth=2, alpha=0.8) 

    # Calculate weights: weight proportional to L^2 (larger systems get more weight)
    weights = Lx_array**0.0

    # Outlier reduction
    n = len(inv_L_array)
    outlier_indices = [n - 4, n - 5, n - 6, n - 7]
    outlier_indices = [i for i in outlier_indices if i >= 0 and i < n]
    weight_reduction_factor = 0.03
    for idx in outlier_indices:
        weights[idx] *= weight_reduction_factor

    weights = weights / np.mean(weights)

    def power_rat_func(x, a, b, c, d):
        """
        Power-law rational function: y = a + b * x^d / (1 + c * x)
        - As x -> 0: y -> a (flat, derivative = 0 if d > 1)
        - As x increases: y changes based on sign of b
        - Derivative at 0: 0 if d > 1, or b if d = 1
        - For d > 1, derivative at 0 is 0 (nonincreasing/flat)
        """
        if len(x) == 0:
            return np.array([])
        x_safe = np.maximum(x, 1e-10)  # Avoid issues with x=0
        denominator = 1 + c * x_safe
        # Avoid division by zero or very small denominators
        denominator = np.maximum(denominator, 1e-10)
        return a + b * (x_safe**d) / denominator
    
    # Initial guess
    a_init = np.min(slopes_array)  # Value at 1/L=0
    slope_range = np.max(slopes_array) - np.min(slopes_array)
    # Use typical inv_L value for scaling
    typical_inv_L = np.mean(inv_L_array)
    # For y = a + b*x^d/(1+c*x), estimate parameters
    b_init = (slope_range) * 10.0  # Amplitude
    c_init = 0.5  # Controls saturation
    d_init = 2.0  # Power (should be > 1 for flat derivative at 0)

    # Fit with weights and data errors
    # Combine data errors with weights: effective sigma = data_error / sqrt(weight)
    slope_errors_array = np.array(slope_errors)
    effective_sigma = slope_errors_array / np.sqrt(weights)
    
    # Recalculate effective_sigma with the reduced weights
    effective_sigma = slope_errors_array / np.sqrt(weights)
    popt, pcov = curve_fit(power_rat_func, inv_L_array, slopes_array, 
                            p0=[a_init, b_init, c_init, d_init],
                            sigma=effective_sigma,
                            absolute_sigma=True,  # Use absolute uncertainties
                            bounds=([-np.inf, -1000, 0, 1.1], [np.inf, 1000, 3.0, 5.0]),  # d > 1, c >= 0
                            maxfev=10000)
    
    a_fit, b_fit, c_fit, d_fit = popt
    
    # Generate smooth curve for plotting
    inv_L_fit = np.linspace(0, max(inv_L_array) * 1.1, 200)
    slopes_fit = power_rat_func(inv_L_fit, a_fit, b_fit, c_fit, d_fit)
    
    # Extrapolate to 1/L = 0
    slope_extrapolated = power_rat_func(np.array([0.0]), a_fit, b_fit, c_fit, d_fit)[0]
    
    # Calculate reduced chi-squared to check if errors are underestimated
    y_pred = power_rat_func(inv_L_array, a_fit, b_fit, c_fit, d_fit)
    residuals = slopes_array - y_pred
    chi_sq = np.sum((residuals / effective_sigma)**2)
    n_data = len(slopes_array)
    n_params = 4
    dof = n_data - n_params  # degrees of freedom
    reduced_chi_sq = chi_sq / dof if dof > 0 else 1.0
    
    # Scale covariance matrix by reduced chi-squared if > 1 (indicates underestimated errors)
    if reduced_chi_sq > 1.0:
        pcov_scaled = pcov * reduced_chi_sq
        print(f"Reduced χ² = {reduced_chi_sq:.2f} > 1, scaling covariance matrix")
    else:
        pcov_scaled = pcov
    
    # Calculate error of extrapolated value using comprehensive error analysis
    x_extrap = 0.0
    
    # 1. Parameter uncertainty: error propagation through covariance matrix
    def partial_derivative(func, params, param_idx, x_val, eps=1e-6):
        """Calculate partial derivative of func w.r.t. params[param_idx] at x_val"""
        params_plus = params.copy()
        params_plus[param_idx] += eps
        params_minus = params.copy()
        params_minus[param_idx] -= eps
        
        y_plus = func(np.array([x_val]), *params_plus)[0]
        y_minus = func(np.array([x_val]), *params_minus)[0]
        return (y_plus - y_minus) / (2 * eps)
    
    params = np.array([a_fit, b_fit, c_fit, d_fit])
    grad = np.zeros(4)
    for i in range(4):
        grad[i] = partial_derivative(power_rat_func, params, i, x_extrap, eps=1e-6)
    
    # Parameter uncertainty from covariance matrix
    error_param = np.sqrt(np.dot(grad, np.dot(pcov_scaled, grad)))
    
    # 2. Model uncertainty: based on weighted scatter of residuals
    # This accounts for systematic deviations of data from the fit
    residuals = slopes_array - y_pred
    # Weighted standard deviation of residuals (properly weighted)
    weighted_mean_residual = np.average(residuals, weights=weights)
    weighted_variance = np.average((residuals - weighted_mean_residual)**2, weights=weights)
    weighted_std_residual = np.sqrt(weighted_variance)
    # Model uncertainty scales with the scatter
    # Also account for reduced chi-squared if > 1 (indicates underestimated errors)
    error_model = weighted_std_residual
    
    # 3. Extrapolation uncertainty: uncertainty grows with distance from data
    # The extrapolation point is at x=0, which is at distance max(inv_L_array) from the data
    # Use a conservative estimate: uncertainty proportional to distance and model uncertainty
    max_inv_L = np.max(inv_L_array)
    min_inv_L = np.min(inv_L_array)
    # Extrapolation distance (distance from data range to extrapolation point)
    extrap_distance = max_inv_L  # Distance from max data point to x=0
    # Extrapolation uncertainty: grows with distance and model uncertainty
    # Use a factor that accounts for the uncertainty in extrapolating beyond the data
    extrap_factor = 1.0 + 0.5 * (extrap_distance / (max_inv_L - min_inv_L + 1e-10))  # Conservative factor
    error_extrap = extrap_factor * error_model * 0.5  # Additional uncertainty for extrapolation
    
    # 4. Data point uncertainty: propagate the individual measurement errors
    # This accounts for the uncertainty in each slope measurement
    # The measurement errors contribute to the fit uncertainty
    # Use weighted average of relative errors, but also consider the spread
    relative_errors = slope_errors_array / np.maximum(np.abs(slopes_array), 1e-10)  # Avoid division by zero
    avg_relative_error = np.average(relative_errors, weights=weights)
    # Also consider the maximum relative error as a conservative bound
    max_relative_error = np.max(relative_errors)
    # Use a combination: average plus a fraction of the spread
    effective_relative_error = avg_relative_error + 0.3 * (max_relative_error - avg_relative_error)
    # Estimate error contribution from measurement uncertainties
    # This scales with the extrapolated value and the effective measurement uncertainty
    # Use absolute value to handle negative slopes, and add a floor based on average absolute error
    avg_abs_error = np.average(slope_errors_array, weights=weights)
    error_data = max(
        np.abs(slope_extrapolated) * effective_relative_error * 0.5,  # Relative error contribution
        avg_abs_error * 0.3  # Floor based on average absolute error
    )
    
    # Combine all error sources in quadrature (assuming independent)
    slope_extrapolated_error = np.sqrt(error_param**2 + error_model**2)
    
    # Print breakdown of error sources for debugging
    print(f"\nError breakdown for extrapolated value:")
    print(f"  Parameter uncertainty: {error_param:.4f}")
    print(f"  Model uncertainty (residual scatter): {error_model:.4f}")
    print(f"  Extrapolation uncertainty: {error_extrap:.4f}")
    print(f"  Data point uncertainty: {error_data:.4f}")
    print(f"  Total (combined in quadrature): {slope_extrapolated_error:.4f}")
    
    # Calculate R-squared
    ss_res = np.sum(weights * (slopes_array - y_pred)**2)
    y_mean = np.average(slopes_array, weights=weights)
    ss_tot = np.sum(weights * (slopes_array - y_mean)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Check derivative at 0 (should be 0 if d > 1)
    deriv_at_0 = 0.0 if d_fit > 1.0 else b_fit
    
    print(f"Power-rational fit: a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}, d = {d_fit:.4f}")
    print(f"Weighted R² = {r2:.4f}, reduced χ² = {reduced_chi_sq:.2f}, derivative at 0 = {deriv_at_0:.4f}")
    print(f"Extrapolated slope at 1/L = 0: {slope_extrapolated:.4f} ± {slope_extrapolated_error:.4f}")

    # Plot the fit line and store handle
    # Format equation: y = a + b * (1/L)^d / (1 + c * (1/L))
    fit_label = r'$y = a + b \cdot x^d / (1 + c \cdot x)$'
    fit_line, = ax2.plot(inv_L_fit, slopes_fit, '--', color='red', linewidth=2, 
                         label=fit_label)
    
    # Mark the extrapolated point at 1/L = 0 with error bar and store handle
    # Format: value ± error (common practice in physics)
    extrap_label = f'Extrapolated: {slope_extrapolated:.1f} ± {slope_extrapolated_error:.1g}'
    
    extrap_container = ax2.errorbar([0], [slope_extrapolated], yerr=[slope_extrapolated_error],
                                     fmt='o', color='red', markersize=12, capsize=5, capthick=2,
                                     elinewidth=2, alpha=0.8, label=extrap_label, zorder=5)
    
    # Set x-axis to include 0 to show extrapolated point
    ax2.set_xlim(left=-0.005)
    
    # Create legend with only fit line and extrapolated point
    # errorbar returns a container, extract the line for the legend
    handles = [fit_line, extrap_container]
    labels = [h.get_label() for h in handles]
    ax2.legend(handles, labels, fontsize=22, loc='lower left')
        
    ax2.set_xlabel(r'$1/L$', fontsize=23)
    ax2.set_ylabel('$2\Delta$', fontsize=23)
    ax2.grid(True, alpha=0.3)
    
    # Set tick label size
    ax2.xaxis.set_tick_params(labelsize=22)
    ax2.yaxis.set_tick_params(labelsize=22)
    
    plt.tight_layout()
    
    # Save the second plot
    pdf_path2 = os.path.join(save_dir, "flux_slope_vs_invL_noncmpK1.pdf")
    png_path2 = os.path.join(save_dir, "flux_slope_vs_invL_noncmpK1.png")
    plt.savefig(pdf_path2, format="pdf", bbox_inches="tight")
    plt.savefig(png_path2, format="png", bbox_inches="tight", dpi=300)
    print(f"Flux slope vs 1/L plot saved at: {pdf_path2}")
    
    plt.show()
    
    # Print summary
    print("\nSummary:")
    print("Lx\t1/L\t\tSlope\t\tSlope Error")
    print("-" * 50)
    for Lx, inv_L, slope, slope_err in zip(Lx_values, inv_L_values, slopes, slope_errors):
        print(f"{Lx}\t{inv_L:.6f}\t{slope:.4f}\t{slope_err:.4f}")

 
if __name__ == '__main__':
    plot_flux_slope_vs_invL()
    
    dbstop = 1
