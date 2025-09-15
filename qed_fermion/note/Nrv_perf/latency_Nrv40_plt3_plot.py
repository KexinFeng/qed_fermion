from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

try:
    from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
    set_default_plotting()
except Exception:
    pass


def _power_law(x, a, b):
    return a * np.power(x, b)


def plot(ax: plt.Axes):
    # Data (copied from original plotting script)
    L = [10, 16, 20, 30, 36, 40, 46, 50]
    se_latency = [0.13, 0.29, 0.91, 5.91, 11.75, 16.89, 32.15, 42.25]
    mc_latency = [0.58, 0.85, 1.40, 2.98, 4.78, 5.64, 9.12, 11.79]

    L_cubed = [l**3 for l in L]

    popt_mc, _ = curve_fit(_power_law, L_cubed, mc_latency)
    coeff_mc, exponent_mc = popt_mc
    x_fit = np.logspace(np.log10(600), np.log10(4e5), 200)

    popt_se, _ = curve_fit(_power_law, L_cubed, se_latency)
    coeff_se, exponent_se = popt_se

    line1, = ax.plot(L_cubed, mc_latency, f'C{0}o', label='HQMC Sampling')
    fit_line_mc, = ax.plot(x_fit, _power_law(x_fit, coeff_mc, exponent_mc), f'C{0}-',
                           label=f'$y \\sim (L^3)^{{{exponent_mc:.3f}}}$')

    line3, = ax.plot(L_cubed, se_latency, f'C{1}o', label='HQMC Measurement')
    fit_line_se, = ax.plot(x_fit, _power_law(x_fit, coeff_se, exponent_se), f'C{1}-',
                           label=f'$y \\sim (L^3)^{{{exponent_se:.3f}}}$')

    # Guideline
    ref_x = L_cubed[0]
    x_fit_guideline = x_fit
    ref_y = 0.38
    scaling = 7/3
    guideline = ref_y * (np.array(x_fit_guideline) / ref_x) ** scaling
    line2, = ax.plot(x_fit_guideline, guideline, f'C{2}--', label=r'DQMC complexity: $L^{7}$')

    lines = [line1, line3, line2, fit_line_mc, fit_line_se]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper left', ncol=1)

    ax.set_xlabel(r'$L^3$')
    ax.set_ylabel('Latency (s / sample)')
    ax.set_xlim(6e2, 4e5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Enforce sparse, publication-style log y-ticks even when saving to PDF
    try:
        # ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=6)))
        ax.yaxis.set_minor_locator(mtick.NullLocator())
        ticks = [1e-1, 1e1, 1e3, 1e5]
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(mtick.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))
        ax.yaxis.set_minor_locator(mtick.NullLocator())
    except Exception:
        pass

    # Keep original behavior: do not set ylim, let autoscale decide

    return ax


__all__ = ["plot"]


