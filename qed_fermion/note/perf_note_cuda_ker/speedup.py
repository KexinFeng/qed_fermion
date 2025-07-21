import numpy as np
import matplotlib.pyplot as plt
import os
# import matplotlib
# matplotlib.use('Agg') # write plots to disk without requiring a display or GUI.
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()
from matplotlib.ticker import MaxNLocator


plt.ion()

Ls = [6, 10, 16, 20]
# Ls = [L**3 * 40 for L in Ls]
Ls = [L**3 *40 for L in Ls]
latency_naive_gpu = [8.05, 13.52, 24.43, 30.09]
latency_opt_cuda = [2.71, 4.36, 8.41, 7.82]
speedup = [latency_naive_gpu[i] / latency_opt_cuda[i] for i in range(len(latency_naive_gpu))]   

fig, ax1 = plt.subplots()

# Plot latencies on the primary y-axis (blue)
line1, = ax1.plot(Ls, latency_naive_gpu, marker='o', linestyle='-', color="#2f89e4", label='Naive GPU implementation')
line2, = ax1.plot(Ls, latency_opt_cuda, marker='o', linestyle='-', color="#0A5197", label='Optimized CUDA kernel')
ax1.set_xlabel(r'$V_s \times N_\tau$')
ax1.set_ylabel(r'Latency (sec / sample)')
ax1.tick_params(axis='y')
# ax1.grid(True)
ax1.spines['left'].set_color("#1673d1")  # Set left axis line color
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(False)
# ax1.tick_params(axis='y', colors='#2f89e4')  # Set left y-axis tick colors
ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))

# Create a secondary y-axis for speedup (red)
ax2 = ax1.twinx()
line3, = ax2.plot(Ls, speedup, marker='^', linestyle='--', color='r', label='Speedup ratio')
ax2.set_ylabel(r'Speedup ratio')
ax2.tick_params(axis='y')
ax2.set_ylim([1, 6])

ax2.spines['right'].set_color('r')  # Set right axis line color
ax2.spines['right'].set_linewidth(1.2)
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)  # ← Hide overlapping right spine from ax1
# ax2.tick_params(axis='y', colors='r')  # Set left y-axis tick colors
ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))

# Combine legends from both axes
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

plt.xscale('log')


plt.show(block=False)

# Save the plot
script_path = os.path.dirname(os.path.abspath(__file__))
class_name = __file__.split('/')[-1].replace('.py', '')
method_name = "latency_speedup"
save_dir = os.path.join(script_path, f"./figures/{class_name}")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, f"{method_name}.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")


