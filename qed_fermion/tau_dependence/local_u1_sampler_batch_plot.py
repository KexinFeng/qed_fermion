import json
import os
import time
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
script_path = os.path.dirname(os.path.abspath(__file__))

from qed_fermion.coupling_mat2 import initialize_coupling_mat
from qed_fermion.hmc_sampler import HmcSampler
import math
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"device: {device}")

colors = plt.cm.tab20(range(20))

def load_visualize_final_greens(ord, Lsize=(20, 20, 20), step=1000001):
    """
    Visualize green functions with error bar
    """
    # Load numerical data
    data_folder_num = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/"

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    num_site = Lx*Ly*Ltau
    # step = 1600000
    swp = math.ceil(step / num_site)
    filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    res = torch.load(filename)

    G_list = res['G_list']
    G_avg = res['G_avg']
    x = np.array(list(range(G_avg.size(-1))))
    start, end = 100, 800
    seq_idx = np.arange(start * num_site, end * num_site, 2*num_site)

    ## Plot
    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label=f'{Lx}_{Ly}_{Ltau}_mc', color=colors[ord*2], lw=2)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={swp}")
    plt.legend()

    # save_plot
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens"
    save_dir = os.path.join(script_path, f"figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


def load_visualize_final_greens_loglog(ord, Lsize=(20, 20, 20), step=1000001):
    """
    Visualize green functions with error bar
    """
    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    num_site = Lx*Ly*Ltau
    swp = math.ceil(step / num_site)

    # Load numerical data
    data_folder_num = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/"

    filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    res = torch.load(filename)

    G_list = res['G_list']
    G_avg = res['G_avg']
    x = np.array(list(range(G_avg.size(-1))))


    # ----- Load and plot analytical data ----
    data_folder_anal = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
    name = f"N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{Ltau}_num_{Ltau}"
    pkl_filepath = os.path.join(data_folder_anal, f"{name}.pkl")

    with open(pkl_filepath, 'r') as f:
        loaded_data_anal = json.load(f)

    G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1]

    ## Plot analytical
    plt.plot(x+1, G_mean_anal, linestyle='--', marker='*', label=f'{Lx}_{Ly}_{Ltau}_anal', color=colors[ord*2+1], lw=2, ms=10)


    ## -------- Plot numerical ---------
    start, end = 100, 800
    seq_idx = np.arange(start * num_site, end * num_site, 2*num_site)
    # seq_idx = np.arange(start * num_site, end * num_site, int(num_site//10))

    G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))

    # Scale G_mc to match G_anal based on their first elements
    idx_ref = 1
    scale_factor = G_mean_anal[idx_ref] / G_mean[idx_ref]
    G_mean *= scale_factor

    ## Plot numerical
    plt.errorbar(x+1, G_mean, yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label=f'{Lx}_{Ly}_{Ltau}_mc', color=colors[ord*2], lw=2)
    

## Plotting
plt.figure()
points = [(20, 20, 20, 5989464), (10, 10, 20, int(16e5))]
for ord, point in enumerate(points):
    Lx, Ly, Ltau, step = point

    load_visualize_final_greens_loglog(ord, (Lx, Ly, Ltau), step)

# Add labels and title

plt.xscale('log')
plt.yscale('log')

##--------- Plot fit ----------
# length = Ltau
# fit = np.array([0.123 / ((tau - 1)**3 + 0.19) for tau in range(length + 1)])
# plt.loglog(fit, linestyle='-', label='Fit')

plt.xlabel(r'$\tau$')
plt.ylabel('log10(G)')
plt.title(f"Nswp={700}")
plt.legend(title='Lx_Ly_Ltau')

# save_plot
class_name = __file__.split('/')[-1].replace('.py', '')
method_name = "greens_loglog"
save_dir = os.path.join(script_path, f"figure_{class_name}")
os.makedirs(save_dir, exist_ok=True) 
file_path = os.path.join(save_dir, f"{method_name}_{Lx}.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")


points = [(20, 20, 20, 5989464), (10, 10, 20, int(16e5))]
for ord, point in enumerate(points):
    Lx, Ly, Ltau, step = point
    load_visualize_final_greens(ord, (Lx, Ly, Ltau), step)
