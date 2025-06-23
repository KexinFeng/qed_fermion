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
sys.path.insert(0, script_path + '/../../../../')
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator

from load_write2file_convert import time_execution


@time_execution
def plot_spsm(Lsize=(6, 6, 10), bs=5, ipair=0):
    Lx, Ly, Ltau = Lsize

    i1 = ipair
    i2 = ipair + 1
    i3 = ipair + 2

    Js = [1, 1.5, 2, 2.3, 2.5, 3]
    r_afm_values = []
    r_afm_errors = []
    spin_order_values = []
    spin_order_errors = []
    
    vs = Lx**2
    
    # plt.figure(figsize=(8, 6))
    
    for J in Js:
        # Initialize data collection arrays for this J value
        all_data = []
        
        # Calculate the number of parts
        num_parts = math.ceil((end_dqmc - start_dqmc) / part_size)
        
        # Loop through all batches and parts
        for bid in range(bs):
            # if not bid == 0: continue
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
        data = data.reshape(bs, -1, vs, 4)
        # data has shape [num_sample, vs, 4], where the last dim has entries: kx, ky, val, error. 
        # [num_sample]

        # Slice
        data = data[:, start:, :, :]

        r_afm = 1 - data[..., 1, 2] / data[..., 0, 2]

        # spin order
        spin_order = np.mean(data[..., 0, 2], axis=1)
        spin_order_err = np.mean(np.abs(data[..., 0, 3]), axis=1)
        spin_order_values.append(spin_order)
        spin_order_errors.append(spin_order_err)

        # r_afm = spin_order
        rtol = data[:, :, :, 3] / data[:, :, :, 2]
        r_afm_err = abs(rtol[:, :, 0] - rtol[:, :, 1]) * (1 - r_afm)
        
        # Calculate mean and error for plotting
        r_afm_mean = np.mean(r_afm, axis=1)
        r_afm_error = np.mean(r_afm_err, axis=1)
        
        r_afm_values.append(r_afm_mean)
        r_afm_errors.append(r_afm_error)


    # ========== Spin order ========= #
    # Stack r_afm_values and r_afm_errors to arrays of shape [Js_num, bs]
    r_afm_values = np.stack(r_afm_values, axis=0)  # shape: [Js_num, bs]
    r_afm_errors = np.stack(r_afm_errors, axis=0)  # shape: [Js_num, bs]

    # Plot the batch mean
    plt.errorbar(Js, r_afm_values[:, 1], yerr=r_afm_errors[:, 1],
                 linestyle='-', marker='o', lw=2, color=f'C{i1}', label=f'hmc_{Lx}x{Ltau}')
    

    # ---- Load and plot spsm_k.pt mean ---- #
    output_dir = os.path.join(script_path, f"data_se_start{start}/Lx_{Lx}_Ltau_{Ltau}_Nrv_{Nrv}_mxitr_{mxitr}")
    spsm_k_file = os.path.join(output_dir, "spsm_k.pt")
    spsm_k_res = torch.load(spsm_k_file, weights_only=False) # [J/bs, Ly, Lx]
    # Compute afm as 1 - spsm_k_mean[0,0]/spsm_k_mean[0,1] (mimic r_afm definition)
    # Here, we assume spsm_k_res['mean'] has shape [J/bs, Ly, Lx] and [0,0] and [0,1] are the relevant k-points
    spsm_k_mean = spsm_k_res['mean']
    spsm_k_std = spsm_k_res['std']
    # Calculate afm ratio as in r_afm
    afm_vals = 1 - spsm_k_mean[:, 0, 1] / spsm_k_mean[:, 0, 0]
    # Error propagation for ratio: err = |A/B| * sqrt((errA/A)^2 + (errB/B)^2)
    errA = spsm_k_std[:, 0, 1]
    errB = spsm_k_std[:, 0, 0]
    A = spsm_k_mean[:, 0, 1]
    B = spsm_k_mean[:, 0, 0]
    afm_errs = np.abs(A/B) * np.sqrt((errA/A)**2 + (errB/B)**2)
    plt.errorbar(Js, afm_vals, yerr=afm_errs, 
                 fmt='o', color=f'C{i2}', linestyle='-', label=f'se_{Lx}x{Ltau}')

    # ---- Load and plot spsm_k.pt mean groundtruth ---- #
    output_dir = os.path.join(script_path, f"data_inv_start{start}/Lx_{Lx}_Ltau_{Ltau}_Nrv_{Nrv}_mxitr_{mxitr}")
    spsm_k_file = os.path.join(output_dir, "spsm_k.pt")
    spsm_k_res = torch.load(spsm_k_file, weights_only=False) # [J/bs, Ly, Lx]
    spsm_k_mean = spsm_k_res['mean']
    spsm_k_std = spsm_k_res['std']
    A = spsm_k_mean[:, 0, 1]
    B = spsm_k_mean[:, 0, 0]
    errA = spsm_k_std[:, 0, 1]
    errB = spsm_k_std[:, 0, 0]
    afm_vals = 1 - A / B
    afm_errs = np.abs(A/B) * np.sqrt((errA/A)**2 + (errB/B)**2)
    plt.errorbar(Js, afm_vals, yerr=afm_errs, 
                 fmt='o', color=f'C{i3}', linestyle='-', label=f'inv_{Lx}x{Ltau}')


if __name__ == '__main__':
    batch_size = 2
    Nrv = 100
    mxitr = 400

    start = -50  # <--- add this line

    plt.figure(figsize=(8, 6))
    for idx, Lx in enumerate([6, 8]):
        Ltau = Lx * 10

        asym = Ltau / Lx * 0.1

        part_size = 500
        start_dqmc = 5000
        end_dqmc = 10000

        root_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run6_{Lx}_{Ltau}/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_kexin_hk_avg/"

        plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size, ipair=3*idx)
        dbstop = 1

    # Plot setting
    plt.xlabel('J/t', fontsize=14)
    plt.ylabel('afm', fontsize=14)
    # plt.title(f'spin_order vs J LxLtau={Lx}x{Ltau}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show(block=False)
    # Save plot
    method_name = "afm"
    save_dir = os.path.join(script_path, f"./figures_start{start}/afm")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_Nrv{Nrv}_mxitr{mxitr}.pdf")    
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")




