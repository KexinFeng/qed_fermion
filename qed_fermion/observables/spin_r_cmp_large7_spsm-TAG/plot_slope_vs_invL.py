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
data_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/cmp_large4_Nrv40/hmc_check_point_cmp_large4_Nrv40/"

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
            # Pattern with optional Nrv (some files don't have Nrv_*)
            pattern = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_*_bs*_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_*_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5*_cmp_True_step_*.pt"
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
    
    
    # Save the first plot
    save_dir = os.path.join(script_path, "./figures/slope_vs_invL")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path1 = os.path.join(save_dir, "corr_with_fits_cmpK1.pdf")
    png_path1 = os.path.join(save_dir, "corr_with_fits_cmpK1.png")
    plt.savefig(pdf_path1, format="pdf", bbox_inches="tight")
    plt.savefig(png_path1, format="png", bbox_inches="tight", dpi=300)
    print(f"Correlation plot with fits saved at: {pdf_path1}")
    
    plt.show()
    
    # ------------------------------------------------------------ #
    # Second figure: slope vs 1/L
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    
    ax2.errorbar(inv_L_values, slopes, yerr=slope_errors, 
                marker='o', markersize=10, linestyle='', linewidth=2,
                capsize=5, capthick=2, elinewidth=2, alpha=0.8)
    
    # Fit with a function that flattens as 1/L -> 0 (nonincreasing)
    # Using a rational function: y = a + b * (1/L) / (1 + c * (1/L))
    # This naturally flattens to y = a as 1/L -> 0, with derivative at 0 = b
    # To ensure nonincreasing, we constrain b <= 0
    inv_L_array = np.array(inv_L_values)
    slopes_array = np.array(slopes)
    Lx_array = np.array(Lx_values)
    
    # Calculate weights: weight proportional to L^2 (larger systems get more weight)
    weights = Lx_array**0.0
    n = len(weights)
    indices_to_double = [n - 5]
    for idx in indices_to_double:
        if 0 <= idx < n:
            weights[idx] *= 0.5
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

    # Fit with weights and data errors
    # Combine data errors with weights: effective sigma = data_error / sqrt(weight)
    # This gives more weight to larger systems while still accounting for measurement errors
    slope_errors_array = np.array(slope_errors)
    effective_sigma = slope_errors_array / np.sqrt(weights)
    
    # Fit with proper error weighting
    popt, pcov = curve_fit(power_rat_func, inv_L_array, slopes_array, 
                            p0=[a_init, b_init, c_init, d_init],
                            sigma=effective_sigma,
                            absolute_sigma=True,  # Use absolute uncertainties
                            bounds=([-np.inf, -np.inf, 0, 1.1], [np.inf, np.inf, np.inf, 5.0]))  # d > 1, c >= 0
    
    a_fit, b_fit, c_fit, d_fit = popt
    
    # Generate smooth curve for plotting
    inv_L_fit = np.linspace(0, max(inv_L_array) * 1.1, 200)
    slopes_fit = power_rat_func(inv_L_fit, a_fit, b_fit, c_fit, d_fit)
    
    # Extrapolate to 1/L = 0
    slope_extrapolated = power_rat_func(np.array([0.0]), a_fit, b_fit, c_fit, d_fit)[0]
    
    # Calculate error of extrapolated value using error propagation
    # For y = a + b * x^d / (1 + c * x), at x = 0: y = a (since x^d = 0 for d > 1)
    # Error propagation: σ_y² = Σ_i Σ_j (∂y/∂p_i) * (∂y/∂p_j) * cov(p_i, p_j)
    # At x = 0: ∂y/∂a = 1, ∂y/∂b = 0, ∂y/∂c = 0, ∂y/∂d = 0 (for d > 1)
    # So σ_y² = cov(a, a) = pcov[0, 0]
    # But to be general, we calculate partial derivatives at x = 0
    
    x_extrap = 0.0
    # Calculate partial derivatives at x = 0
    # For y = a + b * x^d / (1 + c * x)
    # ∂y/∂a = 1
    # ∂y/∂b = x^d / (1 + c*x) = 0 at x=0 (for d > 1)
    # ∂y/∂c = -b * x^(d+1) / (1 + c*x)^2 = 0 at x=0 (for d > 1)
    # ∂y/∂d = b * x^d * ln(x) / (1 + c*x) = 0 at x=0 (for d > 1, but ln(0) is problematic)
    
    # For numerical stability, use a small epsilon
    eps = 1e-6
    x_eps = eps
    
    # Calculate partial derivatives numerically
    def partial_derivative(func, params, param_idx, x_val, eps=1e-6):
        """Calculate partial derivative of func w.r.t. params[param_idx] at x_val"""
        params_plus = params.copy()
        params_plus[param_idx] += eps
        params_minus = params.copy()
        params_minus[param_idx] -= eps
        
        y_plus = func(np.array([x_val]), *params_plus)[0]
        y_minus = func(np.array([x_val]), *params_minus)[0]
        return (y_plus - y_minus) / (2 * eps)
    
    # Calculate reduced chi-squared to check if errors are underestimated
    y_pred = power_rat_func(inv_L_array, a_fit, b_fit, c_fit, d_fit)
    residuals = slopes_array - y_pred
    chi_sq = np.sum((residuals / effective_sigma)**2)
    n_data = len(slopes_array)
    n_params = 4
    dof = n_data - n_params  # degrees of freedom
    reduced_chi_sq = chi_sq / dof if dof > 0 else 1.0
    
    # Scale covariance matrix by reduced chi-squared if > 1 (indicates underestimated errors)
    # This is common practice when errors might be underestimated
    if reduced_chi_sq > 1.0:
        pcov_scaled = pcov * reduced_chi_sq
        print(f"Reduced χ² = {reduced_chi_sq:.2f} > 1, scaling covariance matrix")
    else:
        pcov_scaled = pcov
    
    params = np.array([a_fit, b_fit, c_fit, d_fit])
    grad = np.zeros(4)
    for i in range(4):
        grad[i] = partial_derivative(power_rat_func, params, i, x_extrap, eps=1e-6)
    
    # Comprehensive error analysis combining multiple sources of uncertainty
    
    # 1. Parameter uncertainty from covariance matrix
    error_param = np.sqrt(np.dot(grad, np.dot(pcov_scaled, grad)))
    
    # 2. Model uncertainty: based on weighted scatter of residuals
    # This accounts for systematic deviations of data from the fit
    y_pred = power_rat_func(inv_L_array, a_fit, b_fit, c_fit, d_fit)
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
    
    # Also calculate error from parameter a alone (at x=0, y = a for d > 1)
    # This serves as a check - should be similar to the full error propagation
    slope_extrapolated_error_a = np.sqrt(pcov_scaled[0, 0])
    
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
    print(f"Weighted R² = {r2:.4f}, reduced χ² = {reduced_chi_sq:.2f}, derivative at 0 = {deriv_at_0:.4f}")
    print(f"Extrapolated slope at 1/L = 0: {slope_extrapolated:.4f} ± {slope_extrapolated_error:.4f}")

    # Plot the fit line and store handle
    # Format equation: y = a + b * (1/L)^d / (1 + c * (1/L))
    fit_label = r'$y = y_0 + \frac{b \cdot x^d}{1 + c \cdot x}$'
    fit_line, = ax2.plot(inv_L_fit, slopes_fit, '--', color='red', linewidth=2, 
                         label=fit_label)
    
    # Mark the extrapolated point at 1/L = 0 with error bar and store handle
    # Format: value ± error (common practice in physics)
    extrap_label = f'{slope_extrapolated:.1f} ± {slope_extrapolated_error:.2f}'
    
    extrap_container = ax2.errorbar([0], [slope_extrapolated], yerr=[slope_extrapolated_error],
                                     fmt='o', color='red', markersize=12, capsize=5, capthick=2,
                                     elinewidth=2, alpha=0.8,label=extrap_label, zorder=5)
    
    # Set x-axis to include 0 to show extrapolated point
    ax2.set_xlim(left=-0.005)
    
    # Create legend with only fit line and extrapolated point
    # errorbar returns a container, extract the line for the legend
    handles = [fit_line, extrap_container]  # extrap_container[0] is the line/marker
    labels = [h.get_label() for h in handles]
    ax2.legend(handles, labels, fontsize=22, loc='upper left')
        
    ax2.set_xlabel(r'$1/L$', fontsize=23)
    ax2.set_ylabel('$2\Delta$', fontsize=23)
    ax2.grid(True, alpha=0.3)
    
    # Set tick label size
    ax2.xaxis.set_tick_params(labelsize=22)
    ax2.yaxis.set_tick_params(labelsize=22)
    
    plt.tight_layout()
    
    # Save the second plot
    pdf_path2 = os.path.join(save_dir, "slope_vs_invL_cmpK1.pdf")
    png_path2 = os.path.join(save_dir, "slope_vs_invL_cmpK1.png")
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

