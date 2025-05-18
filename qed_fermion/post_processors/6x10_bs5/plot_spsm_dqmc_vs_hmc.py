import re
import time
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

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator

from load_write2file_convert import time_execution

# Add partition parameters
part_size = 500
start_dqmc = 2000
end_dqmc = 6000
root_folder = "/Users/kx/Desktop/forked/dqmc_u1sl_mag/run4/"
# root_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/data6810/hmc_check_point"

@time_execution
def plot_spsm(Lsize=(6, 6, 10), bs=5):
    Js = [0.5, 1.0, 3.0]
    r_afm_values = []
    r_afm_errors = []
    spin_order_values = []
    spin_order_errors = []
    
    Lx, Ly, Ltau = Lsize
    vs = Lx**2
    
    # plt.figure(figsize=(8, 6))
    
    for J in Js:
        # Initialize data collection arrays for this J value
        all_data = []
        
        # Calculate the number of parts
        num_parts = math.ceil((end_dqmc - start_dqmc) / part_size)
        
        # Loop through all batches and parts
        for bid in range(bs):
            for part_id in range(num_parts):
                input_folder = root_folder + f"/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}_bid{bid}_part_{part_id}_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"
                name = f"spsm.bin"
                ftdqmc_filename = os.path.join(input_folder, name)
                
                try:
                    part_data = np.genfromtxt(ftdqmc_filename)
                    all_data.append(part_data)
                    print(f'Loaded ftdqmc data: {ftdqmc_filename}')
                except (FileNotFoundError, ValueError) as e:
                    raise RuntimeError(f'Error loading {ftdqmc_filename}: {str(e)}') from e
        
        # Combine all parts' data
        data = np.concatenate(all_data)
        data = data.reshape(-1, vs, 4)
        # data has shape [num_sample, vs, 4], where the last dim has entries: kx, ky, val, error. 
        # [num_sample]
        r_afm = 1 - data[:, 1, 2] / data[:, 0, 2]

        # spin order
        spin_order = np.mean(data[:, 0, 2])
        spin_order_err = np.mean(np.abs(data[:, 0, 3]))
        spin_order_values.append(spin_order)
        spin_order_errors.append(spin_order_err)

        # r_afm = spin_order
        rtol = data[:, :, 3] / data[:, :, 2]
        r_afm_err = abs(rtol[:, 0] - rtol[:, 1]) * (1 - r_afm)
        
        # Calculate mean and error for plotting
        r_afm_mean = np.mean(r_afm)
        r_afm_error = np.mean(r_afm_err)
        
        r_afm_values.append(r_afm_mean)
        r_afm_errors.append(r_afm_error)

        # # ---------- color map of spin order ---------- #
        # data_mean = data.mean(axis=0)

        # # Visualize data_mean as a color map
        # x = data_mean[:, 0]
        # y = data_mean[:, 1]
        # values = data_mean[:, 2]

        # plt.figure(figsize=(8, 6))
        # plt.tricontourf(x, y, values, levels=100, cmap='viridis')
        # plt.colorbar(label=f'J= {J:.2g}')
        # plt.xlabel('kx', fontsize=14)
        # plt.ylabel('ky', fontsize=14)
        # plt.title(f'Mean Data Visualization J= {J:.2g}', fontsize=14)
        # plt.grid(True, alpha=0.3)

    # ========== AFM ========= #
    plt.figure(figsize=(8, 6))
    # Plot the errorbar for the means
    plt.errorbar(Js, r_afm_values, yerr=r_afm_errors, 
                linestyle='-', marker='o', lw=2, color='blue', label='hmc_r_afm')
    
    # Load dqmc and plot
    dqmc_filename = "/Users/kx/Desktop/hmc/benchmark_dqmc/piflux_B0.0K1.0_L6_tuneJ_kexin_hk_avg/tuning_js_sectune_l6_spin_coratio.dat"
    data = np.genfromtxt(dqmc_filename)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], 
                 fmt='o', color='red', linestyle='-', label='dqmc_r_afm')
    
    # Plot setting
    plt.xlabel('J', fontsize=14)
    plt.ylabel('r_afm', fontsize=14)
    plt.title(f'r_afm vs J LxLtau={Lx}x{Ltau}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # save plot
    method_name = "spsm"
    save_dir = os.path.join(script_path, f"./figures/r_afm")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_LxLtau_{Lx}x{Ltau}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    # ========== Spin order ========= #
    plt.figure(figsize=(8, 6))
    plt.errorbar(Js, spin_order_values, yerr=spin_order_errors, 
                linestyle='-', marker='o', lw=2, color='blue', label='hmc_spin_order')
    
    # Load dqmc and plot
    dqmc_filename = "/Users/kx/Desktop/hmc/benchmark_dqmc/piflux_B0.0K1.0_L6_tuneJ_kexin_hk_avg/tuning_js_sectune_l6_spin_order.dat"
    data = np.genfromtxt(dqmc_filename)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], 
                 fmt='o', color='red', linestyle='-', label='dqmc_spin_order')
    
    # Plot setting
    plt.xlabel('J', fontsize=14)
    plt.ylabel('spin_order', fontsize=14)
    plt.title(f'spin_order vs J LxLtau={Lx}x{Ltau}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # save plot
    method_name = "spin_order"
    save_dir = os.path.join(script_path, f"./figures/spin_order")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_L{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    return r_afm_values, r_afm_errors


if __name__ == '__main__':
    Lx = 6
    Ltau = 10
    batch_size = 5  
    plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size)
    plt.show(block=True)



