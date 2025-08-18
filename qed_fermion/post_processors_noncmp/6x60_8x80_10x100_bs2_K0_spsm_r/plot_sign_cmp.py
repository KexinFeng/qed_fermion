import matplotlib.pyplot as plt

plt.ion()

import numpy as np
import math

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import selective_log_label_func, set_default_plotting
set_default_plotting()

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator

# Add partition parameters

# Lx = int(os.getenv("Lx", '6'))
# print(f"Lx: {Lx}")
# Ltau = int(os.getenv("Ltau", '60'))
# print(f"Ltau: {Ltau}")

# asym = Ltau / Lx * 0.1

# part_size = 500
# start_dqmc = 5000
# end_dqmc = 10000

load_postprcessed_data = False

def plot_spsm(Lsize=(6, 6, 10), bs=5, ipair=(0, 1)):
    Lx, Ly, Ltau = Lsize

    i1, i2 = ipair

    Js = [1, 1.5, 2, 2.3, 2.5, 3]

    # ---- Load dqmc and plot ----
    dqmc_filename = dqmc_folder + f"/tuning_js_sectune_l{Lx}_sgn_single.dat"
    data = np.genfromtxt(dqmc_filename)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], 
                 fmt='o', color=f'C{i2}', linestyle='-', label=fr'${Ltau}x{Lx}^2$ DQMC')
    dbstop = 1
    

if __name__ == '__main__':
    batch_size = 2

    plt.figure()
    for idx, Lx in enumerate([6, 8, 10]):
        Ltau = Lx * 10

        asym = Ltau / Lx * 0.1

        part_size = 500
        start_dqmc = 5000
        end_dqmc = 10000

        root_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run6_{Lx}_{Ltau}/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_kexin_hk_avg/"
        # dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810_nc/piflux_B0.0K0.0_tuneJ_b1l_noncompact_kexin_hk/sgn_single"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810_nc/piflux_B0.0K0.0_tuneJ_b1l_noncompact_kexin_hk_avg/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/dqmc_data/noncmp_benchmark_det_sign/piflux_B0.0K1.0_tuneJ_b1l_kexin_hk_avg/"

        plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size, ipair=(2*idx, 2*idx + 1))
        dbstop = 1

    # Plot setting
    plt.xlabel(r'$J$')
    plt.ylabel(r'$\langle sign(\det M) \rangle$')
    # plt.title(f'spin_order vs J LxLtau={Lx}x{Ltau}', fontsize=16)
    # plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show(block=False)
    # Save plot
    method_name = "sign_cmp"
    save_dir = os.path.join(script_path, f"./figures/sign_cmp")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")




