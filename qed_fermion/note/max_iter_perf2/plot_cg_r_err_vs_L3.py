from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

# Add import for plot style
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
set_default_plotting()
plt.ion()

import re
import torch

# Directory containing the checkpoint files
ckpt_dir = os.path.join(os.path.dirname(__file__), 'hmc_check_point_pcg_iter')

# Parse all filenames
all_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]

# Extract L, max_iter, step from filenames
pattern = re.compile(r'Ltau_(\d+)_Nstp_(\d+)_bs1_Jtau_1.2_K_0_dtau_0.1_.*?_max_iter_(\d+)_.*?_step_(\d+)\.pt')

# Organize files by (L, max_iter) -> step
file_dict = {}
for fname in all_files:
    m = pattern.search(fname)
    if m:
        Ltau, Nstp, max_iter, step = map(int, m.groups())
        L = Ltau // 10
        key = (L, max_iter)
        step = int(step)
        if key not in file_dict or step > file_dict[key][0]:
            file_dict[key] = (step, fname)

# Now, for each max_iter, collect (L, mean_log10_err)
max_iters = sorted(set(k[1] for k in file_dict.keys()))
L_list = sorted(set(k[0] for k in file_dict.keys()))

results = {mi: [] for mi in max_iters}

for (L, max_iter), (step, fname) in file_dict.items():
    if max_iter == 1200: continue
    print(f"Processing {fname} with \nmax_iter={max_iter} and L={L}")

    fpath = os.path.join(ckpt_dir, fname)
    d = torch.load(fpath, map_location='cpu')
    cg_r_err = d['cg_r_err_list']  # shape (N_step, 1)
    # Take last 2000 steps
    cg_r_err_last = cg_r_err[-2000:]
    mean_err = cg_r_err_last.mean().item()
    sigma_err = cg_r_err_last.std().item() / np.sqrt(len(cg_r_err_last))
    results[max_iter].append((L, mean_err, sigma_err))

# Plot log10(mean_err) vs L^3 for each max_iter
fig = plt.figure()
lines = []
labels = []
for idx, max_iter in enumerate(sorted(results.keys())):
    arr = np.array(sorted(results[max_iter]))  # shape (n, 2)
    if arr.size == 0:
        continue
    Ls = arr[:,0]
    val_errs = arr[:,1]
    sigma_errs = arr[:,2]
    line = plt.errorbar(Ls, val_errs, yerr=sigma_errs, fmt=f'C{idx}o-', label=f'max_iter={max_iter}')
    lines.append(line)
    labels.append(line.get_label())

plt.xlabel(r'$L$')
plt.ylabel(r'$\log_{10} err$')
# plt.title('log10(CG residual error) vs $L^3$ for each max_iter')
plt.legend(lines, labels, loc='best')
plt.grid(True)
plt.xscale('linear')
plt.yscale('log', base=10)

# ax = fig.gca()
# ax.yaxis.set_major_formatter(FuncFormatter(selective_log_label_func(ax, numticks=6)))

# Save the plot
save_dir = os.path.join(script_path, 'figures')
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, 'cg_r_err_vs_L3.pdf')
plt.savefig(file_path, format='pdf', bbox_inches='tight')
print(f"Figure saved at: {file_path}")
plt.show() 