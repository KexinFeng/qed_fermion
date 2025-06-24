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

use_dqmc_post_processor = False

@time_execution
def plot_spsm(Lsize=(6, 6, 10), bs=5, ipair=0):
    Lx, Ly, Ltau = Lsize

    i1, i2, i3 = ipair*3, ipair*3 + 1, ipair*3 + 2

    Js = [1, 1.5, 2, 2.3, 2.5, 3]
    r_afm_values = []
    r_afm_errors = []
    spin_order_values = []
    spin_order_errors = []
    
    vs = Lx**2
    
    # plt.figure(figsize=(8, 6))
    if use_dqmc_post_processor:
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
        # plt.errorbar(Js, spin_order_values, yerr=spin_order_errors, 
        #             linestyle='-', marker='o', lw=2, color='blue', label='hmc_spin_order')
        
        # Stack spin_order_values and spin_order_errors to arrays of shape [Js_num, bs]
        spin_order_values = np.stack(spin_order_values, axis=0)  # shape: [Js_num, bs]
        spin_order_errors = np.stack(spin_order_errors, axis=0)  # shape: [Js_num, bs]

        # Filter out outliers in spin_order_values (values > 100)
        spin_order_values = np.where(spin_order_values > 10, np.nan, spin_order_values)

        # Plot the batch mean
        plt.errorbar(Js, spin_order_values[:, 1] / vs, yerr=spin_order_errors[:, 1] / vs, linestyle='-', marker='o', lw=2, color=f'C{i1}', label=f'hmc_{Lx}x{Ltau}')

    # ---- Load dqmc and plot ----
    filename = dqmc_folder + f"/tuning_js_sectune_l{Lx}_spin_order.dat"
    data = np.genfromtxt(filename)
    plt.errorbar(data[:, 0], data[:, 1] / vs, yerr=data[:, 2] / vs, 
                 fmt='o', color=f'C{i2}', linestyle='-', label=f'dqmc_{Lx}x{Ltau}', alpha=0.6)
    
    # ---- Load hmc and plot ----
    xs = []
    ys = []
    yerrs = []
    for J in Js:
        filename = hmc_folder + f"/ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_{J}_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_False_step_10000.pt"
        data = torch.load(filename, map_location='cpu')
        spsm_k_list = data['spsm_k_list'][start_dqmc:end_dqmc]
        ys.append(spsm_k_list.mean(axis=(0, 1))[0, 0])
        yerrs.append(
            std_root_n(spsm_k_list.numpy(), axis=0, lag_sum=200)[0, 0, 0]
        )

        # Test spsm_k lag_sum, which is at least 5000
        gamma0 = std_root_n(spsm_k_list.numpy(), axis=0, lag_sum=1)[1] * spsm_k_list.shape[0]**0.5
        total_err = init_convex_seq_estimator(data['spsm_k_list'][:, 1])
        print(f'J={J}, lag_sum: {(total_err/gamma0)**2}')
        
        xs.append(J)

    plt.errorbar(np.array(xs), np.array(ys) / vs, yerr=np.array(yerrs)/ vs, 
                 fmt='o', color=f'C{i3}', linestyle='-', label=f'hmcse_{Lx}x{Ltau}', alpha=0.6)


if __name__ == '__main__':
    batch_size = 2

    plt.figure(figsize=(8, 6))
    for idx, Lx in enumerate([6, 8, 10]):
        Ltau = Lx * 10

        asym = int(Ltau / Lx * 0.1)

        part_size = 500
        start_dqmc = 5000
        end_dqmc = 10000

        hmc_folder = f"/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmp_6810/hmc_check_point_noncmp_bench1/"
        root_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run6_{Lx}_{Ltau}_noncmp_K1/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810_nc/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_noncompact_kexin_hk_avg/"

        plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size, ipair=idx)
        dbstop = 1

    # Plot setting
    plt.xlabel('J/t', fontsize=14)
    plt.ylabel('S_AF / Ns', fontsize=14)
    # plt.title(f'spin_order vs J LxLtau={Lx}x{Ltau}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    
    plt.show(block=False)
    # Save plot
    method_name = "spin_order"
    save_dir = os.path.join(script_path, f"./figures/spin_order")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")




