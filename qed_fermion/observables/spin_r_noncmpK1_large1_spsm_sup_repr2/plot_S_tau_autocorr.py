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
    pos_idx = np.where(autocorr_pos > 0)[0]
    if len(pos_idx) == 0 or pos_idx[0] != 0:
        return np.nan, np.nan, None

    first_nonpos = np.where(autocorr_pos <= 0)[0]
    end = first_nonpos[0] if len(first_nonpos) > 0 else len(autocorr_pos)
    k_fit = lags_pos[:end]
    autocorr_fit = autocorr_pos[:end]
    if len(k_fit) < 5:
        return np.nan, np.nan, None

    A_guess = np.max(autocorr_fit)
    tau_guess = k_fit[-1] / 3.0
    offset_guess = np.min(autocorr_fit)

    try:
        popt, pcov = curve_fit(
            exponential_decay,
            k_fit,
            autocorr_fit,
            p0=[A_guess, tau_guess, offset_guess],
            bounds=([0, 1, -0.1], [2.0, len(k_fit), 0.1]),
            maxfev=5000,
        )
        tau = popt[1]
        tau_err = np.sqrt(pcov[1, 1]) if not np.isnan(pcov[1, 1]) else np.nan
        return tau, tau_err, (popt[0], popt[1], popt[2])
    except Exception:
        return np.nan, np.nan, None


def plot_S_tau_autocorr():
    """Compute and plot autocorrelation-k curves for S_tau density across lattice sizes."""

    lattice_sizes = [10, 12, 16, 20, 30, 36, 40, 46, 56, 60]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.33))

    corr_lengths = {}
    corr_lengths_err = {}

    for i, Lx in enumerate(lattice_sizes):
        Ltau = int(10 * Lx)

        import glob

        def find_hmc_file(Lx, Ltau, folder=data_folder):
            pattern = (
                f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_*_bs*_Jtau_1.2_K_1_dtau_0.1_"
                f"delta_0.028_N_leapfrog_5_m_1_cg_rtol_*_max_block_idx_1_gear0_steps_1000_"
                f"dt_deque_max_len_5_Nrv_*_cmp_False_step_*.pt"
            )
            files = glob.glob(os.path.join(folder, pattern))
            if not files:
                print(f"No file found for Lx={Lx}, Ltau={Ltau} in {folder}")
                return None

            def extract_step(filename):
                m = re.search(r'step_(\d+)\\.pt', filename)
                return int(m.group(1)) if m else 0

            files.sort(key=extract_step, reverse=True)
            return files[0]

        hmc_filename = find_hmc_file(Lx, Ltau)
        if hmc_filename is None:
            continue

        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')

        # S_tau time series: [timesteps, batch_size]
        S_tau = res['S_tau_list']

        # Average over batch and compute density
        S_tau_avg = S_tau.mean(axis=1).cpu().numpy()
        volume = Lx * Lx * Ltau
        S_tau_density = np.abs(S_tau_avg) / volume

        # Equilibrium detection — mirror S_plaq logic with late-window mean
        if Lx >= 30:
            late_window = S_tau_density[5000:6000] if len(S_tau_density) >= 6000 else S_tau_density[max(0, len(S_tau_density)-1000):]
        else:
            late_window = S_tau_density[3000:4000] if len(S_tau_density) >= 4000 else S_tau_density[max(0, len(S_tau_density)-800):]
        if len(late_window) == 0:
            print(f'Lx={Lx}: insufficient data for equilibrium detection')
            continue
        EQUILIBRIUM_THRESHOLD = late_window.mean() * 0.99
        equilibrium_indices = np.where(S_tau_density <= EQUILIBRIUM_THRESHOLD)[0]
        if len(equilibrium_indices) == 0:
            print(f'Lx={Lx}: No equilibrium point found (density never reached {EQUILIBRIUM_THRESHOLD})')
            continue

        equilibrium_start_idx = equilibrium_indices[0]
        equilibrium_data = S_tau_density[equilibrium_start_idx:]

        print(
            f'Lx={Lx}: Equilibrium {EQUILIBRIUM_THRESHOLD:.3g} starts at step {equilibrium_start_idx}, '
            f'using {len(equilibrium_data)} equilibrium points'
        )

        # Autocorrelation
        lags, autocorr = compute_autocorrelation(equilibrium_data)
        mask = lags <= 3000
        lags = lags[mask]
        autocorr = autocorr[mask]

        if len(lags) == 0 or len(autocorr) == 0:
            print(f'Lx={Lx}: Failed to compute autocorrelation')
            continue

        # Fit tau
        tau, tau_err, fit_params = fit_autocorr_length(lags, autocorr)
        if not np.isnan(tau) and Lx <= 40:
            corr_lengths[Lx] = tau
            corr_lengths_err[Lx] = tau_err
            print(f'Lx={Lx}: tau_L={tau:.2f} ± {tau_err:.2f}')

        # Plot autocorr curve (sparse points)
        max_plot_points = 500
        if len(lags) > max_plot_points:
            subsample_step = len(lags) // max_plot_points
            plot_indices = np.arange(0, len(lags), subsample_step)
            lags_plot = lags[plot_indices]
            autocorr_plot = autocorr[plot_indices]
        else:
            lags_plot = lags
            autocorr_plot = autocorr

        ax.plot(lags_plot, autocorr_plot, 'o', alpha=1.0, markersize=4, linewidth=1.0, label=fr'${Ltau}\times{Lx}^2$')

        # Plot exponential fit if available
        if not np.isnan(tau) and fit_params is not None:
            mask = (0 < lags) & (autocorr > 0)
            if np.sum(mask) > 0:
                k_min = lags[mask].min()
                k_max = lags[mask].max()
                k_fit_smooth = np.linspace(k_min, k_max, 200)
                A_fit, tau_fit, offset_fit = fit_params
                autocorr_fit_values = exponential_decay(k_fit_smooth, A_fit, tau_fit, offset_fit)
                mask_positive = autocorr_fit_values > 0
                if np.sum(mask_positive) > 0:
                    ax.plot(
                        k_fit_smooth[mask_positive],
                        autocorr_fit_values[mask_positive],
                        '-',
                        alpha=1.0,
                        linewidth=1.0,
                        color=ax.lines[-1].get_color(),
                    )

    ax.set_xlabel("Lag $k$", fontsize=14)
    ax.set_ylabel("$S_{\\tau}$ autocorrelation", fontsize=14)
    ax.set_xlim(left=0, right=300)
    ax.set_ylim(bottom=-0.4)
    ax.legend(fontsize=11, ncol=3, loc='lower left')
    ax.grid(True, alpha=0.3, which='both')

    # Inset: tau_L vs L
    if len(corr_lengths) > 0:
        Lx_sorted = sorted(corr_lengths.keys())
        tau_values = [corr_lengths[Lx] for Lx in Lx_sorted]
        tau_errors = [corr_lengths_err.get(Lx, 0) for Lx in Lx_sorted]

        inset_width = 0.58
        inset_height = 0.58
        inset_ax = inset_axes(
            ax,
            width=f"{inset_width*100}%", height=f"{inset_height*100}%",
            bbox_to_anchor=(0.35, 0.4, inset_width, inset_height),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        inset_ax.errorbar(
            Lx_sorted,
            tau_values,
            yerr=tau_errors,
            fmt='k^',
            linewidth=2,
            markersize=6,
            capsize=4,
        )
        inset_ax.set_xlabel("$L$", fontsize=13)
        inset_ax.set_ylabel("$\\tau_L$", fontsize=13)
        inset_ax.grid(True, alpha=0.3)
        inset_ax.tick_params(axis='both', which='major', labelsize=13)
        inset_ax.xaxis.set_tick_params(labelsize=13)
        inset_ax.yaxis.set_tick_params(labelsize=13)

        # Optional: power-law fit tau_L ~ L^z
        def power_law(L, a, z):
            return a * L**z
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
                    sigma=(tau_err_arr[valid] if len(tau_err_arr) == len(Lx_arr) and np.all(tau_err_arr[valid] > 0) else None),
                    absolute_sigma=(True if len(tau_err_arr) == len(Lx_arr) and np.all(tau_err_arr[valid] > 0) else False),
                )
                a_fit, z_fit = popt
                err_a, err_z = np.sqrt(np.diag(pcov))
                Lx_fit = np.linspace(min(Lx_arr[valid]), max(Lx_arr[valid]), 200)
                tau_fit = power_law(Lx_fit, a_fit, z_fit)
                inset_ax.plot(Lx_fit, tau_fit, 'b--', lw=2, label=fr"$\sim L^{{{z_fit:.2f}}}$")
                inset_ax.text(0.05, 0.9, fr"$z = {z_fit:.2f} \pm {err_z:.2f}$", transform=inset_ax.transAxes,
                              fontsize=12, verticalalignment='top', color='b')
                inset_ax.legend(fontsize=11, frameon=False)
            except Exception as e:
                print("Power-law fit failed:", e)

    # Panel label
    ax.text(-0.13, 0.98, "(a)", transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(script_path, "./figures/S_tau_autocorr")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "S_tau_autocorr_noncmpK1.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show()


if __name__ == '__main__':
    plot_S_tau_autocorr()
