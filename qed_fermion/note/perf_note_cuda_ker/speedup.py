import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import rcParams
rcParams['figure.raise_window'] = False
from matplotlib.ticker import MaxNLocator
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()

def extract_latency_from_line(line):
    m = re.search(r'\[.*?([\d\.]+)s/it', line)
    if m:
        return float(m.group(1))
    m = re.search(r'\[.*?([\d\.]+)it/s', line)
    if m:
        return 1.0 / float(m.group(1))
    return None

def get_middle_latency(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    tqdm_lines = [l for l in lines if re.search(r'\d+%\|', l) and ('it/s' in l or 's/it' in l)]
    if not tqdm_lines:
        return None
    mid = len(tqdm_lines) // 2
    latency = extract_latency_from_line(tqdm_lines[mid])
    return latency

def get_L_from_filename(filename):
    m = re.search(r'_L(\d+)_', filename)
    if m:
        return int(m.group(1))
    return None

base_dir = os.path.dirname(os.path.abspath(__file__))
err_dir = os.path.join(base_dir, 'report_latency_cuda_kernel')
files = os.listdir(err_dir)

ck0 = {}
ck1 = {}
for fname in files:
    if fname.endswith('.err'):
        L = get_L_from_filename(fname)
        if L is not None:
            if fname.startswith('ck0'):
                ck0[L] = fname
            elif fname.startswith('ck1'):
                ck1[L] = fname

Ls_all = sorted(set(ck0.keys()) | set(ck1.keys()))
latency_ck0 = []
latency_ck1 = []
Ls_speedup = []
speedup = []
Ls_ck1_only = []
latency_ck1_only = []

for L in sorted(ck1.keys()):
    f1 = os.path.join(err_dir, ck1[L])
    lat1 = get_middle_latency(f1)
    if lat1 is not None:
        if L in ck0:
            f0 = os.path.join(err_dir, ck0[L])
            lat0 = get_middle_latency(f0)
            if lat0 is not None:
                latency_ck0.append(lat0)
                latency_ck1.append(lat1)
                Ls_speedup.append(L)
                speedup.append(lat0 / lat1 if lat1 else np.nan)
            else:
                # kernel off OOM, but kernel on is valid
                Ls_ck1_only.append(L)
                latency_ck1_only.append(lat1)
        else:
            # No kernel off at all, but kernel on is valid
            Ls_ck1_only.append(L)
            latency_ck1_only.append(lat1)

fig, ax1 = plt.subplots()

if Ls_speedup:
    line1, = ax1.plot(np.array(Ls_speedup)**3*10, latency_ck0, marker='o', linestyle='-', color="#2f89e4", label='Kernel off')
    line2, = ax1.plot(np.array(Ls_speedup)**3*10, latency_ck1, marker='o', linestyle='-', color="#0A5197", label='Kernel on (matched)')
# if Ls_ck1_only:
#     line2_only, = ax1.plot(np.array(Ls_ck1_only)**3*10, latency_ck1_only, marker='x', linestyle='None', color="#0A5197", label='Kernel on (no off/OOM)')

ax1.set_xlabel(r'$V_s\times N_\tau$')
ax1.set_ylabel(r'Latency (s / sample)')
ax1.tick_params(axis='y')
ax1.spines['left'].set_color("#1673d1")
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Only plot speedup where both are available
if Ls_speedup:
    ax2 = ax1.twinx()
    line3, = ax2.plot(np.array(Ls_speedup)**3*10, speedup, marker='^', linestyle='--', color='r', label='Latency ratio')
    ax2.set_ylabel(r'Latency ratio')
    ax2.tick_params(axis='y')
    ax2.set_ylim([1, max(6, np.nanmax(speedup)*1.1)])
    ax2.spines['right'].set_color('r')
    ax2.spines['right'].set_linewidth(1.2)
    ax2.spines['right'].set_visible(True)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    lines = [line1, line2, line3] if Ls_ck1_only else [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
else:
    lines = [line2_only]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.65, 1.0))

plt.xscale('log')
plt.show(block=False)

save_dir = os.path.join(base_dir, "figures/speedup")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "latency_speedup_cuda_kernel.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}") 