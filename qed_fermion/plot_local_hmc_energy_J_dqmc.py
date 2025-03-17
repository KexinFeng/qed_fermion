import re
import matplotlib.pyplot as plt

plt.ion()

import numpy as np

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator


def plot_energy_J(Js=[], starts=[500], sample_steps=[1]):
    Js = [0.5, 1, 3]
    xs = Js

    Sb_plaq_list_hmc = []
    Sb_plaq_list_local = []
    Stau_list_hmc = []
    Stau_list_local = []

    for J in Js:

        hmc_filename = f"/Users/kx/Desktop/hmc/fignote/local_vs_hmc_check_fermion/hmc_check_point/ckpt_N_hmc_6_Ltau_10_Nstp_10000_bs5_Jtau_{J:.1g}_K_1_dtau_0.1_step_10000.pt"
        local_update_filename = f"/Users/kx/Desktop/hmc/fignote/local_vs_hmc_check_fermion/local_check_point/ckpt_N_local_6_Ltau_10_Nstp_3600000bs_5_Jtau_{J:.1g}_K_1_dtau_0.1_step_3600000.pt"

        res = torch.load(hmc_filename)
        print(f'Loaded: {hmc_filename}')
        Sb_plaq_list_hmc.append(res['S_plaq_list'])
        Stau_list_hmc.append(res['S_tau_list'])

        res = torch.load(local_update_filename)
        print(f'Loaded: {local_update_filename}')
        Sb_plaq_list_local.append(res['S_plaq_list'])
        Stau_list_local.append(res['S_tau_list'])

    # ====== Index ====== #
    hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    end = int(hmc_match.group(1))
    start = starts.pop(0)
    sample_step = sample_steps.pop(0)
    seq_idx = np.arange(start, end, sample_step)
    seq_idx_init = np.arange(0, end, sample_step)

    local_match = re.search(r'Nstp_(\d+)', local_update_filename)
    end_local = int(local_match.group(1))
    start_local = starts.pop(0)
    sample_step_local = sample_steps.pop(0)
    seq_idx_local = np.arange(start_local, end_local, sample_step_local)
    seq_idx_local_init = np.arange(0, end_local, sample_step_local)

    # ======= Plot Sb ======= #
    plt.figure()

    # HMC
    ys = [Sb_plaq[seq_idx].mean().item() for Sb_plaq in Sb_plaq_list_hmc]  # [seq, bs]
    yerr1 = [error_mean(init_convex_seq_estimator(Sb_plaq[seq_idx_init].numpy()) / np.sqrt(seq_idx_init.size)) * 1.96 for Sb_plaq in Sb_plaq_list_hmc]
    # yerr1 = [std_root_n(Sb_plaq[seq_idx].numpy(), axis=0, lag_sum=100).mean() for Sb_plaq in Sb_plaq_list_hmc]
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
        
    # Local
    ys = [Sb_plaq[seq_idx_local].mean().item() for Sb_plaq in Sb_plaq_list_local]  # [seq, bs]
    yerr1 = [error_mean(init_convex_seq_estimator(Sb_plaq[seq_idx_local_init].numpy()) /np.sqrt(seq_idx_local_init.size)) * 1.96 for Sb_plaq in Sb_plaq_list_local] 
    # yerr1 = [std_root_n(Sb_plaq[seq_idx_local].numpy(), axis=0, lag_sum=400).mean() for Sb_plaq in Sb_plaq_list_local]
    yerr2 = [t_based_error(Sb_plaq[seq_idx_local].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_local] 
    print(yerr1, '\n', yerr2)
    yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    # yerr = np.sqrt(np.array(yerr1)**2)
    plt.errorbar(xs, ys, yerr=yerr, linestyle='-', marker='*', markersize=10, lw=2, color='green', label='local')
    for idx, bi in enumerate(range(Sb_plaq_list_local[0].size(1))):
        ys = [Sb_plaq[seq_idx_local, bi].mean().item() for Sb_plaq in Sb_plaq_list_local] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='*', markersize=10, lw=2, color=f"C{idx}")

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
    yerr1 = [error_mean(init_convex_seq_estimator(Stau[seq_idx_init].numpy()) / np.sqrt(seq_idx_init.size)) * 1.96 for Stau in Stau_list_hmc]
    # yerr1 = [std_root_n(Stau[seq_idx].numpy(), axis=0, lag_sum=100).mean() for Stau in Stau_list_hmc]
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

    # Local
    ys = [Stau[seq_idx_local].mean().item() for Stau in Stau_list_local]  # [seq, bs]
    yerr1 = [error_mean(init_convex_seq_estimator(Stau[seq_idx_local_init].numpy()) / np.sqrt(seq_idx_local_init.size)) * 1.96  for Stau in Stau_list_local]  
    # yerr1 = [std_root_n(Stau[seq_idx_local].numpy(), axis=0, lag_sum=400).mean() for Stau in Stau_list_local]
    yerr2 = [t_based_error(Stau[seq_idx_local].mean(axis=0).numpy()) for Stau in Stau_list_local] 
    print(yerr1, '\n', yerr2)
    yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    # yerr = np.sqrt(np.array(yerr1)**2)
    plt.errorbar(xs, ys, yerr=yerr, linestyle='-', marker='*', markersize=10, lw=2, color='green', label='local')
    for idx, bi in enumerate(range(Stau_list_local[0].size(1))):
        ys = [Stau[seq_idx_local, bi].mean().item() for Stau in Stau_list_local] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='*', markersize=10, lw=2, color=f"C{idx}")

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
    Lx, Ly, Ltau = 6, 6, 10
    Vs = Lx * Ly * Ltau

    plot_energy_J(starts=[3000, 3000*Vs], sample_steps=[1, Vs])

    dbstop = 1


