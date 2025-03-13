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
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.stat import t_based_error


def plot_energy_J(Js=[], starts=[500], sample_steps=[1]):
    Js = [0.5]
    xs = Js

    Sb_plaq_list_hmc = []
    Sb_plaq_list_local = []
    Sf_list_hmc = []
    Sf_list_local = []

    for J in Js:

        hmc_filename = script_path + "/check_points/hmc_check_point/ckpt_N_hmc_6_Ltau_10_Nstp_10000_Jtau_0.5_K_1_dtau_0.1_step_10000.pt"
        local_update_filename = script_path + "/check_points/local_check_point/ckpt_N_local_6_Ltau_10_Nstp_3600000_Jtau_0.5_K_1_dtau_0.1_step_3600000.pt"

        res = torch.load(hmc_filename)
        print(f'Loaded: {hmc_filename}')
        Sb_plaq_list_hmc.append(res['S_plaq_list'])
        Sf_list_hmc.append(res['S_tau_list'])

        res = torch.load(local_update_filename)
        print(f'Loaded: {local_update_filename}')
        Sb_plaq_list_local.append(res['S_plaq_list'])
        Sf_list_local.append(res['S_tau_list'])

    # ====== Index ====== #
    hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    end = int(hmc_match.group(1))
    start = starts.pop(0)
    sample_step = sample_steps.pop(0)
    seq_idx = np.arange(start, end, sample_step)

    local_match = re.search(r'Nstp_(\d+)', local_update_filename)
    end_local = int(local_match.group(1))
    start_local = starts.pop(0)
    sample_step_local = sample_steps.pop(0)
    seq_idx_local = np.arange(start_local, end_local, sample_step_local)
  

    # ======= Plot Sb ======= #
    plt.figure()

    # HMC
    ys = [Sb_plaq[seq_idx].mean().item() for Sb_plaq in Sb_plaq_list_hmc]  # [seq, bs]
    # yerr_s = [std_root_n(Sb_plaq[seq_idx].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_hmc] 
    yerr_s = [t_based_error(Sb_plaq[seq_idx].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_hmc] 
    plt.errorbar(xs, ys, yerr=yerr_s, linestyle='-', marker='o', lw=2, color='blue', label='hmc')
    for bi in range(Sb_plaq_list_hmc[0].size(1)):
        ys = [Sb_plaq[seq_idx, bi].mean().item() for Sb_plaq in Sb_plaq_list_hmc] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='o', lw=2)
           
    # Local
    ys = [Sb_plaq[seq_idx_local].mean().item() for Sb_plaq in Sb_plaq_list_local]  # [seq, bs]
    # yerr_s = [std_root_n(Sb_plaq[seq_idx_local].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_local]
    yerr_s = [t_based_error(Sb_plaq[seq_idx_local].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_local]
    plt.errorbar(xs, ys, yerr=yerr_s, linestyle='-', marker='*', ms=8, lw=2, color='green', label='local')
    for bi in range(Sb_plaq_list_local[0].size(1)):
        ys = [Sb_plaq[seq_idx_local, bi].mean().item() for Sb_plaq in Sb_plaq_list_local] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='*', ms=8, lw=2)

    plt.xlabel(r"$J$")
    plt.ylabel(r"$S_{plaq}$")
    plt.legend()

    # save plot
    method_name = "boson"
    save_dir = os.path.join(script_path, f"./figures/energies_boson")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    # ====== Plot Sf ======= #
    plt.figure()
    # HMC
    ys = [Sf[seq_idx].mean().item() for Sf in Sf_list_hmc]  # [seq, bs]
    # yerr_s = [std_root_n(Sf[seq_idx].mean(axis=0).numpy()) for Sf in Sf_list_hmc]
    yerr_s = [t_based_error(Sf[seq_idx].mean(axis=0).numpy()) for Sf in Sf_list_hmc]
    plt.errorbar(xs, ys, yerr=yerr_s, linestyle='-', marker='o', lw=2, color='blue', label='hmc')
    for bi in range(Sf_list_hmc[0].size(1)):
        ys = [Sf[seq_idx, bi].mean().item() for Sf in Sf_list_hmc] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='o', lw=2)

    # Local
    ys = [Sf[seq_idx_local].mean().item() for Sf in Sf_list_local]  # [seq, bs]
    # yerr_s = [std_root_n(Sf[seq_idx_local].mean(axis=0).numpy()) for Sf in Sf_list_local]
    yerr_s = [t_based_error(Sf[seq_idx_local].mean(axis=0).numpy()) for Sf in Sf_list_local]
    plt.errorbar(xs, ys, yerr=yerr_s, linestyle='-', marker='*', ms=8, lw=2, color='green', label='local')
    for bi in range(Sf_list_local[0].size(1)):
        ys = [Sf[seq_idx_local, bi].mean().item() for Sf in Sf_list_local] 
        plt.errorbar(
            xs, 
            ys, 
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='*', ms=8, lw=2)

    plt.xlabel(r"$J$")
    plt.ylabel(r"$S_{plaq}$")
    plt.legend()

    plt.xlabel(r"$J$")
    plt.ylabel(r"$-log(detM)$")
    plt.legend()

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

    plot_energy_J(starts=[3000, 2000*Vs], sample_steps=[1, Vs])

    dbstop = 1


