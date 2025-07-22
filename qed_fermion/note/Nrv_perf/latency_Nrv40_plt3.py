import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

# Add import for curve_fit
from scipy.optimize import curve_fit

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()
plt.ion()

# Data
L = [10, 16, 20, 30, 36, 40, 46, 50]  # beta = L
se_latency = [0.13, 0.29, 0.91, 5.91, 11.75, 16.89, 32.15, 42.25]  # sec/item
mc_latency = [0.58, 0.85, 1.40, 2.98, 4.78, 5.64, 9.12, 11.79]  # sec/item
se_footage = [326.00, 552.00, 820.00, 2096.00, 3416.00, 4552.00, 7416.00, 8644.00]  # MB
mc_footage = [1683.69, 1949.69, 2145.69, 2647.69, 2979.69, 3187.69, 3547.69, 3791.69] # MB

# Convert L to L^3
L_cubed = [l**3 for l in L]

# --- Fit MC latency to power law ---
def power_law(x, a, b):
    return a * np.power(x, b)

# MC latency fit (blue)
popt_mc, pcov_mc = curve_fit(power_law, L_cubed, mc_latency)
coeff_mc, exponent_mc = popt_mc
alpha_mc = 3 * exponent_mc

# SE latency fit (red)
popt_se, pcov_se = curve_fit(power_law, L_cubed, se_latency)
coeff_se, exponent_se = popt_se
alpha_se = 3 * exponent_se

# Create plot
fig = plt.figure()

# Plot mc_latency (blue)
line1, = plt.plot(L_cubed, mc_latency, f'C{0}o', label='HMC latency')
fit_line_mc, = plt.plot(L_cubed, power_law(np.array(L_cubed), coeff_mc, exponent_mc), f'C{0}-',
                       label=f'$y \sim (L^3)^{{{exponent_mc:.3f}}}$')

# Plot se_latency (red)
line3, = plt.plot(L_cubed, se_latency, f'C{1}o', label='SE latency')
fit_line_se, = plt.plot(L_cubed, power_law(np.array(L_cubed), coeff_se, exponent_se), f'C{1}-',
                       label=f'$y \sim (L^3)^{{{exponent_se:.3f}}}$')

plt.xlabel(r'$L^3$')
plt.ylabel('Latency (s / sample)')

# Add guideline
ref_idx = 0
ref_x = L_cubed[ref_idx]
ref_y = mc_latency[ref_idx] / 3
scaling = 7/3
guideline = ref_y * (np.array(L_cubed) / ref_x) ** scaling
line2, = plt.plot(L_cubed, guideline, f'C{2}--', label=r'$L^{7}$ guideline')

lines = [line1, line3, line2, fit_line_mc, fit_line_se]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='upper left', ncol=2)

plt.grid(True)
plt.xscale('log')
plt.yscale('log')

ax = fig.gca()
ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=4))

# Save the plot
script_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_path, "figures_Nrv40_plt")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "mc_se_latency_vs_L3.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

