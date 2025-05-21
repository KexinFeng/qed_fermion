import math
import re
import time
import matplotlib.pyplot as plt

plt.ion()

import numpy as np

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator

from load_write2file_convert import time_execution

part_size = 500
start_dqmc = 5000
end_dqmc = 10000
end_hmc = 10000

dqmc_dir = "/Users/kx/Desktop/forked/dqmc_u1sl_mag/run5_blk/"
hmc_folder = f"/Users/kx/Desktop/hmc/fignote/equilibrium_issue/hmc_check_point_block_tau/"

@time_execution
def plot_energy_J(Lx, Ltau, Js=[], starts=[500], sample_steps=[1]):

    xs = Js

    Sb_plaq_list_hmc = []
    Sb_plaq_list_dqmc = []

    Stau_list_hmc = []
    Stau_list_dqmc = []

    bs = 2

    for J in Js:
        end = end_hmc
        # hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_{end}_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_lmd_0.99_sig_min_0.8_sig_max_1.2_lower_limit_0.5_upper_limit_0.8_mass_mode_0_step_{end}.pt"
        hmc_file = f"/Users/kx/Desktop/hmc/fignote/equilibrium_issue/hmc_check_point_block_tau/ckpt_N_t_hmc_{Lx}_Ltau_{Ltau}_Nstp_{end}_bs{2}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_lmd_0.99_sig_min_0.8_sig_max_1.2_lower_limit_0.5_upper_limit_0.8_mass_mode_0_step_{end}.pt"

        hmc_filename = os.path.join(hmc_folder, hmc_file)

        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')
        Sb_plaq_list_hmc.append(res['S_plaq_list'])
        Stau_list_hmc.append(res['S_tau_list'])

        # Aggregate DQMC data from all parts
        all_Sb_plaq_data = []
        all_Stau_data = []
        beta = Ltau * 0.1
        N = Lx * Lx * beta
        
        num_parts = math.ceil((end_dqmc - start_dqmc )/ part_size)
        for bid in range(bs):
            for part_id in range(num_parts):
                # dqmc_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run3/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}_part_{part_id}_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"
                dqmc_folder = dqmc_dir + f"/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}_bid{bid}_part_{part_id}_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"

                name = f"ener1.bin"
                dqmc_filename = os.path.join(dqmc_folder, name)
                
                try:
                    data = np.genfromtxt(dqmc_filename).reshape(-1, 15) # 15 is the item number in ener1.bin
                    all_Sb_plaq_data.append(data[:, 3] * N)
                    all_Stau_data.append(data[:, 2] * N)
                    print(f'Loaded DQMC data: {dqmc_filename}')
                except (FileNotFoundError, ValueError) as e:
                    raise RuntimeError(f'Error loading {dqmc_filename}: {str(e)}') from e
        
        # Concatenate data from all parts
        combined_Sb_plaq = np.concatenate(all_Sb_plaq_data).reshape(bs, -1)
        combined_Stau = np.concatenate(all_Stau_data).reshape(bs, -1)
        Sb_plaq_list_dqmc.append(combined_Sb_plaq)
        Stau_list_dqmc.append(combined_Stau)


    # ====== Index ====== #
    hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    end = int(hmc_match.group(1))
    # end = 5000
    start = starts.pop(0)
    sample_step = sample_steps.pop(0)
    seq_idx = np.arange(start, end, sample_step)
    seq_idx_init = np.arange(0, end, sample_step)


    # ======= Plot Sb ======= #
    plt.figure()

    # HMC
    ys = [Sb_plaq[seq_idx].mean().item() for Sb_plaq in Sb_plaq_list_hmc]  # [seq, bs]
    # yerr1 = [error_mean(init_convex_seq_estimator(Sb_plaq[seq_idx_init].numpy()) / np.sqrt(seq_idx_init.size)) * 1.00 for Sb_plaq in Sb_plaq_list_hmc]
    yerr1 = [std_root_n(Sb_plaq[seq_idx].numpy(), axis=0, lag_sum=50).mean() for Sb_plaq in Sb_plaq_list_hmc]
    yerr2 = [t_based_error(Sb_plaq[seq_idx].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_hmc] 
    print(yerr1, '\n', yerr2)
    yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    # yerr = np.sqrt(np.array(yerr1)**2)
    plt.errorbar(xs, ys, yerr=yerr, linestyle='-', marker='o', lw=2, color='blue', label='hmc')
    for idx, bi in enumerate(range(Sb_plaq_list_hmc[0].size(1))):
        ys = [Sb_plaq[seq_idx, bi].mean().item() for Sb_plaq in Sb_plaq_list_hmc] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='o', lw=2, color=f"C{idx}")
        
    # DQMC
    ys = [Sb_plaq.mean() for Sb_plaq in Sb_plaq_list_dqmc]  # [seq, bs]
    # yerr1 = [error_mean(init_convex_seq_estimator(Sb_plaq) / np.sqrt(Sb_plaq.size)) for Sb_plaq in Sb_plaq_list_dqmc]
    yerr1 = [std_root_n(Sb_plaq, axis=0, lag_sum=50).mean() for Sb_plaq in Sb_plaq_list_dqmc]
    # yerr2 = [t_based_error(Sb_plaq.mean(axis=0)) for Sb_plaq in Sb_plaq_list_dqmc]
    # print(yerr1, '\n', yerr2)
    # yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    plt.errorbar(xs, ys, yerr=yerr1, linestyle='-', marker='s', lw=2, color='red', label='dqmc')

    # Plot setting
    plt.xlabel(r"$J$")
    plt.ylabel(r"$S_{plaq}$")
    plt.legend(ncol=2)

    # save plot
    method_name = "boson"
    save_dir = os.path.join(script_path, f"./figures/energies_boson")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


    # ====== Plot Stau ======= #
    plt.figure()
    # HMC
    ys = [Stau[seq_idx].mean().item() for Stau in Stau_list_hmc]  # [seq, bs]
    # yerr1 = [error_mean(init_convex_seq_estimator(Stau[seq_idx_init].numpy()) / np.sqrt(seq_idx_init.size)) * 1.00 for Stau in Stau_list_hmc]
    yerr1 = [std_root_n(Stau[seq_idx].numpy(), axis=0, lag_sum=10).mean() for Stau in Stau_list_hmc]
    yerr2 = [t_based_error(Stau[seq_idx].mean(axis=0).numpy()) for Stau in Stau_list_hmc] 
    print(yerr1, '\n', yerr2)
    yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    # yerr = np.sqrt(np.array(yerr1)**2)
    plt.errorbar(xs, ys, yerr=yerr, linestyle='-', marker='o', lw=2, color='blue', label='hmc')
    for idx, bi in enumerate(range(Stau_list_hmc[0].size(1))):
        ys = [Stau[seq_idx, bi].mean().item() for Stau in Stau_list_hmc] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='o', lw=2, color=f"C{idx}")
       
    # DQMC
    ys = [Stau.mean() for Stau in Stau_list_dqmc]  # [seq, bs]
    # yerr1 = [error_mean(init_convex_seq_estimator(Stau) / np.sqrt(Stau.size)) for Stau in Stau_list_tau]
    yerr1 = [std_root_n(Stau, axis=0, lag_sum=10).mean() for Stau in Stau_list_dqmc]
    # yerr2 = [t_based_error(Stau.mean(axis=0)) for Stau in Stau_list_tau]
    # yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    plt.errorbar(xs, ys, yerr=yerr1, linestyle='-', marker='s', lw=2, color='red', label='dqmc')

    # Plot settings
    plt.xlabel(r"$J$")
    # plt.ylabel(r"$-log(detM)$")
    plt.ylabel(r"$S_{tau}$")
    plt.legend(ncol=2)

    # save plot
    method_name = "fermion"
    save_dir = os.path.join(script_path, f"./figures/energies_fermion")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show(block=True)


if __name__ == '__main__':
    Lx, Ly, Ltau = 6, 6, 240
    # Lx, Ly, Ltau = 6, 6, 10
    Vs = Lx * Ly * Ltau

    Js = [1.0, 1.5, 2.0, 2.5, 3.0]
    Js = [1.0, 2.0, 2.5, 3.0]
    # Js = [2.0, 2.5, 3.0]
    # Js = [0.5, 1.0, 3.0]

    plot_energy_J(Lx, Ltau, Js=Js, starts=[5000], sample_steps=[1])

    dbstop = 1


