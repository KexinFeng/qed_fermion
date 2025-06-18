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

# Lx = int(os.getenv("Lx", '6'))
# print(f"Lx: {Lx}")
# Ltau = int(os.getenv("Ltau", '60'))
# print(f"Ltau: {Ltau}")

# asym = Ltau / Lx * 0.1

# part_size = 500
# start_dqmc = 5000
# end_dqmc = 10000



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

        # # spin order
        # spin_order = np.mean(data[..., 0, 2], axis=1)
        # spin_order_err = np.mean(np.abs(data[..., 0, 3]), axis=1)
        # spin_order_values.append(spin_order)
        # spin_order_errors.append(spin_order_err)

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
    if Lx == 6:
        plt.errorbar(Js, r_afm_values[:, 1], yerr=r_afm_errors[:, 1],
                 linestyle='-', marker='o', lw=2, color=f'C{i1}', label=f'hmc_{Lx}x{Ltau}')
    else:
        plt.errorbar(Js[:-1], r_afm_values[:, 1][:-1], yerr=r_afm_errors[:, 1][:-1],
                 linestyle='-', marker='o', lw=2, color=f'C{i1}', label=f'hmc_{Lx}x{Ltau}')
    
    # ---- Load dqmc and plot ----
    filename = dqmc_folder + f"/tuning_js_sectune_l{Lx}_spin_coratio.dat"
    data = np.genfromtxt(filename)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], 
                 fmt='o', color=f'C{i2}', linestyle='-', label=f'dqmc_{Lx}x{Ltau}')
    
    # ---- Load hmc and plot ----
    xs = []
    ys = []
    yerrs = []
    for J in Js:
        filename = hmc_folder + f"/ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_{J}_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_False_step_10000.pt"
        data = torch.load(filename, map_location='cpu')
        spsm_k_list = data['spsm_k_list'][start_dqmc:end_dqmc]
        spsm_k_mean = spsm_k_list.mean(axis=(0, 1))
        # spsm_k_err = spsm_k_list.err(axis=(0,)).mean(axis=0)

        r_afm = 1 - spsm_k_mean[1, 0] / spsm_k_mean[0, 0]
        # Error propagation for r_afm = 1 - spsm_k_mean[1, 0] / spsm_k_mean[0, 0]
        A = spsm_k_mean[1, 0]
        B = spsm_k_mean[0, 0]
        # Estimate errors using std over samples
        A_err = spsm_k_list[:, 1, 0].std() / np.sqrt(len(spsm_k_list))
        B_err = spsm_k_list[:, 0, 0].std() / np.sqrt(len(spsm_k_list))
        rtol_A = A_err / A if A != 0 else 0
        rtol_B = B_err / B if B != 0 else 0
        r_afm_err = abs(rtol_B - rtol_A) * (1 - r_afm)

        ys.append(r_afm)
        yerrs.append(r_afm_err)
        xs.append(J)

    plt.errorbar(np.array(xs), np.array(ys), yerr=np.array(yerrs), 
                fmt='o', color=f'C{i3}', linestyle='-', label=f'hmcse_{Lx}x{Ltau}')


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
        root_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run6_{Lx}_{Ltau}_noncmp/"
        dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810_nc/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_noncompact_kexin_hk_avg/"

        plot_spsm(Lsize=(Lx, Lx, Ltau), bs=batch_size, ipair=idx)
        dbstop = 1

    # Plot setting
    plt.xlabel('J/t', fontsize=14)
    plt.ylabel('afm', fontsize=14)
    # plt.title(f'spin_order vs J LxLtau={Lx}x{Ltau}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    
    plt.show(block=False)
    # Save plot
    method_name = "afm_ratio"
    save_dir = os.path.join(script_path, f"./figures/afm")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")




