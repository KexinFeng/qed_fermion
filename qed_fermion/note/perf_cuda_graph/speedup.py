from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import rcParams
rcParams['figure.raise_window'] = False
from matplotlib.ticker import MaxNLocator
import os
script_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()

# Helper to extract latency from tqdm lines
# Handles both s/it and it/s

def extract_latency_from_line(line):
    # Try to match [mm:ss<mm:ss,  1.23s/it] or [mm:ss<mm:ss,  1.23it/s]
    m = re.search(r'\[.*?([\d\.]+)s/it', line)
    if m:
        return float(m.group(1))
    m = re.search(r'\[.*?([\d\.]+)it/s', line)
    if m:
        return 1.0 / float(m.group(1))
    return None

def get_most_common_latency(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    tqdm_lines = [l for l in lines if re.search(r'\d+%\|', l) and ('it/s' in l or 's/it' in l)]
    if not tqdm_lines:
        return None
    latencies = []
    for line in tqdm_lines:
        latency = extract_latency_from_line(line)
        if latency is not None:
            latencies.append(round(latency, 3))  # round to 3 decimals for mode
    if not latencies:
        return None
    most_common = Counter(latencies).most_common(1)
    return most_common[0][0] if most_common else None


def get_L_from_filename(filename):
    m = re.search(r'_L(\d+)_', filename)
    if m:
        return int(m.group(1))
    return None

# Gather all .err files
base_dir = os.path.dirname(os.path.abspath(__file__))
err_dir = os.path.join(base_dir, 'report_latency_cudagraph')
files = os.listdir(err_dir)

# Map: L -> (graph0 file, graph1 file)
graph0 = {}
graph1 = {}
for fname in files:
    if fname.endswith('.err'):
        L = get_L_from_filename(fname)
        if L is not None:
            if 'graph0' in fname:
                graph0[L] = fname
            elif 'graph1' in fname:
                graph1[L] = fname

Ls = sorted(set(graph0.keys()) & set(graph1.keys()))
latency_graph0 = []
latency_graph1 = []
for L in Ls:
    f0 = os.path.join(err_dir, graph0[L])
    f1 = os.path.join(err_dir, graph1[L])
    lat0 = get_most_common_latency(f0)
    lat1 = get_most_common_latency(f1)
    latency_graph0.append(lat0)
    latency_graph1.append(lat1)

speedup = [latency_graph0[i] / latency_graph1[i] if latency_graph1[i] else np.nan for i in range(len(Ls))]

fig, ax1 = plt.subplots()

# Plot latencies on the primary y-axis (blue)
line1, = ax1.plot(np.array(Ls), latency_graph0, marker='o', linestyle='-', color="#2f89e4", label='CUDA Graph off')
line2, = ax1.plot(np.array(Ls), latency_graph1, marker='o', linestyle='-', color="#0A5197", label='CUDA Graph on')
ax1.set_xlabel(r'$L$')
ax1.set_ylabel(r'Latency (s / sample)')
ax1.tick_params(axis='y')
ax1.spines['left'].set_color("#1673d1")
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Create a secondary y-axis for speedup (red)
ax2 = ax1.twinx()
line3, = ax2.plot(np.array(Ls), speedup, marker='^', linestyle='--', color='r', label='Latency ratio')
ax2.set_ylabel(r'Latency ratio')
ax2.tick_params(axis='y')
ax2.set_ylim([1, max(6, np.nanmax(speedup)*1.1)])
ax2.spines['right'].set_color('r')
ax2.spines['right'].set_linewidth(1.2)
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Combine legends from both axes
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.65, 1.0))

# plt.xscale('log')
plt.show(block=False)

# Save the plot
save_dir = os.path.join(base_dir, "figures/speedup")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "latency_speedup_cuda_graph.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}") 