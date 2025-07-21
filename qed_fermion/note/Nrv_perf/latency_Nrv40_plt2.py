import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

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

# Create plot
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot mc_latency on left y-axis (blue)
line1, = ax1.plot(L_cubed, mc_latency, 'o-', color="#2f89e4", label='MC latency')
ax1.set_xlabel(r'$L^3$')
ax1.set_ylabel('MC latency (sec/item)', color="#2f89e4")
ax1.tick_params(axis='y', labelcolor="#2f89e4")
ax1.set_yscale('log')

# Add guideline to axis1
ref_idx = 0
ref_x = L_cubed[ref_idx]
ref_y = mc_latency[ref_idx]
scaling = 7/3
guideline = ref_y * (np.array(L_cubed) / ref_x) ** scaling
line2, = ax1.plot(L_cubed, guideline, 'k--', label=r'$L^{7}$ guideline')

# Create a secondary y-axis for se_latency (red)
ax2 = ax1.twinx()
line3, = ax2.plot(L_cubed, se_latency, 's--', color='r', label='SE latency')
ax2.set_ylabel('SE latency (sec/item)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_yscale('log')

# Combine legends from both axes
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

plt.grid(True)
plt.xscale('log')
# plt.yscale('log')

# Save the plot
script_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_path, "figures_Nrv40_plt")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "mc_se_latency_vs_L3.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

