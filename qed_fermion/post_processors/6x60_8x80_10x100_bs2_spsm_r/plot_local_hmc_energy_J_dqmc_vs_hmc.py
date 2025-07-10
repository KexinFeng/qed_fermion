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

Lx = int(os.getenv("Lx", '6'))
print(f"Lx: {Lx}")
Ltau = int(os.getenv("Ltau", '60'))
print(f"Ltau: {Ltau}")
asym = Ltau / Lx * 0.1

end_dqmc = 10000
end_hmc = 10000

hmc_folder = f"/Users/kx/Desktop/hmc/fignote/equilibrium_issue/hmc_check_point_bench/"
dqmc_folder = f"/Users/kx/Desktop/hmc/benchmark_dqmc/L6810/piflux_B0.0K1.0_tuneJ_b{asym:.1g}l_kexin_hk/"

@time_execution
def plot_energy_J(Js=[], starts=[500], sample_steps=[1]):
    # Js = [0.5, 1, 3]
    Js = [1, 1.5, 2, 2.3, 2.5, 3]
    xs = Js

    Sb_plaq_list_hmc = []
    Sb_plaq_list_dqmc = []
    Stau_list_hmc = []
    Stau_list_tau = []

    beta = int(Ltau * 0.1)
    N = Lx * Lx * beta

    bs = 2

    for J in Js:
        # hmc
        # hmc_folder = f"/Users/kx/Desktop/hmc/fignote/ftdqmc/benchmark_6x6x10_bs5/hmc_check_point_6x10"
        # hmc_file = f"ckpt_N_hmc_6_Ltau_10_Nstp_6000_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.05_N_leapfrog_4_m_1_step_6000.pt"

        # hmc_filename = os.path.join(hmc_folder, hmc_file)

        end = end_hmc
        hmc_file = f"ckpt_N_t_hmc_{Lx}_Ltau_{Ltau}_Nstp_{end}_bs{2}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_step_{end}.pt"

        hmc_filename = os.path.join(hmc_folder, hmc_file)

        res = torch.load(hmc_filename, map_location='cpu')
        print(f'Loaded: {hmc_filename}')
        Sb_plaq_list_hmc.append(res['S_plaq_list'])
        Stau_list_hmc.append(res['S_tau_list'])

        # dqmc
        # name_plaq = f"l6b1js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        name_plaq = f"l{Lx}b{Ltau/10:.1g}js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        dqmc_filename_plaq = os.path.join(dqmc_folder + "/ejpi/", name_plaq)
    
        # name_tau = f"l6b1js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        name_tau = f"l{Lx}b{Ltau/10:.1g}js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        dqmc_filename_tau = os.path.join(dqmc_folder + "/ejs/", name_tau)

        data = np.genfromtxt(dqmc_filename_plaq)
        Sb_plaq_list_dqmc.append(data.reshape(-1, 1) * N)
        data = np.genfromtxt(dqmc_filename_tau)
        Stau_list_tau.append(data.reshape(-1, 1) * N)

    # ====== Index ====== #
    hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    end = int(hmc_match.group(1))
    # end = 5000
    start = starts.pop(0)
    sample_step = sample_steps.pop(0)
    seq_idx = np.arange(start, end, sample_step)
    seq_idx_init = np.arange(0, end, sample_step)


    # ======= Plot Sb_plaq ======= #
    plt.figure()

    # HMC
    ys = [Sb_plaq[seq_idx].mean().item() for Sb_plaq in Sb_plaq_list_hmc]  # [seq, bs]
    # yerr1 = [error_mean(init_convex_seq_estimator(Sb_plaq[seq_idx_init].numpy()) / np.sqrt(seq_idx_init.size)) * 1.00 for Sb_plaq in Sb_plaq_list_hmc]
    yerr1 = [std_root_n(Sb_plaq[seq_idx].numpy(), axis=0, lag_sum=50).mean() for Sb_plaq in Sb_plaq_list_hmc]
    yerr2 = [t_based_error(Sb_plaq[seq_idx].mean(axis=0).numpy()) for Sb_plaq in Sb_plaq_list_hmc] 
    # print(yerr1, '\n', yerr2)
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
    # yerr1 = [error_mean(init_convex_seq_estimator(Sb_plaq) / np.sqrt(Sb_plaq.size)) * 1.00 * 36 for Sb_plaq in Sb_plaq_list_dqmc]
    # yerr1 = [std_root_n(Sb_plaq, axis=0, lag_sum=50).mean() for Sb_plaq in Sb_plaq_list_dqmc]
    # yerr2 = [t_based_error(Sb_plaq.mean(axis=0)) for Sb_plaq in Sb_plaq_list_dqmc]
    # print(yerr1, '\n', yerr2)
    # yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    plt.errorbar(xs, ys, linestyle='-', marker='s', lw=2, color='red', label='dqmc')

    # Plot setting
    plt.xlabel(r"$J$")
    plt.ylabel(r"$S_{plaq}$")
    plt.legend(ncol=2)

    # save plot
    method_name = "Splaq"
    save_dir = os.path.join(script_path, f"./figures{Lx}_{Ltau}/energies_Splaq")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    # ====== Plot Stau ======= #
    plt.figure()
    # HMC
    ys = [Stau[seq_idx].mean().item() for Stau in Stau_list_hmc]  # [seq, bs]
    yerr1 = [error_mean(init_convex_seq_estimator(Stau[seq_idx_init].numpy()) / np.sqrt(seq_idx_init.size)) * 1.00 for Stau in Stau_list_hmc]
    # yerr1 = [std_root_n(Stau[seq_idx].numpy(), axis=0, lag_sum=50).mean() for Stau in Stau_list_hmc]
    yerr2 = [t_based_error(Stau[seq_idx].mean(axis=0).numpy()) for Stau in Stau_list_hmc] 
    # print(yerr1, '\n', yerr2)
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
    ys = [Stau.mean() for Stau in Stau_list_tau]  # [seq, bs]
    # yerr1 = [error_mean(init_convex_seq_estimator(Stau) / np.sqrt(Stau.size)) for Stau in Stau_list_tau]
    # yerr2 = [t_based_error(Stau.mean(axis=0)) for Stau in Stau_list_tau]
    # yerr = np.sqrt(np.array(yerr1)**2 + np.array(yerr2)**2)
    plt.errorbar(xs, ys, linestyle='-', marker='s', lw=2, color='red', label='dqmc')

    # Plot settings
    plt.xlabel(r"$J$")
    # plt.ylabel(r"$-log(detM)$")
    plt.ylabel(r"$S_{tau}$")
    plt.legend(ncol=2)

    # save plot
    method_name = "Stau"
    save_dir = os.path.join(script_path, f"./figures{Lx}_{Ltau}/energies_Stau")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show(block=True)


if __name__ == '__main__':
    # Lx, Ly, Ltau = 6, 6, 10
    # Vs = Lx * Ly * Ltau

    plot_energy_J(starts=[5000], sample_steps=[1])

    dbstop = 1


