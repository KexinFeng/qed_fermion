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
sys.path.insert(0, script_path + '/../../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator



def plot_spin_r():
    J = 1.25

    Sb_plaq_list_hmc = []
    Sb_plaq_list_dqmc = []
    Stau_list_hmc = []
    Stau_list_tau = []

    Lx, Ly, Ltau = 10, 10, 100
    beta = int(Ltau * 0.1)
    N = Lx * Lx * beta

    # hmc
    hmc_folder = f"/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/cmp_large/hmc_check_point_large"
    hmc_file = f"/ckpt_N_hmc_10_Ltau_100_Nstp_10000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_True_step_10000.pt"

    hmc_filename = os.path.join(hmc_folder, hmc_file)

    res = torch.load(hmc_filename, map_location='cpu')
    print(f'Loaded: {hmc_filename}')

    




    Sb_plaq_list_hmc.append(res['S_plaq_list'])
    Stau_list_hmc.append(res['S_tau_list'])

    # dqmc
    dqmc_folder = "/Users/kx/Desktop/hmc/benchmark_dqmc/" + "/piflux_B0.0K1.0_L6_tuneJ_kexin_hk/"
    # dqmc_folder = "/Users/kx/Desktop/hmc/benchmark_dqmc/L6b24_avg/piflux_B0.0K1.0_L6b24_tuneJ_kexin_hk/"

    name_plaq = f"l6b1js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
    # name_plaq = f"l6b24js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
    dqmc_filename_plaq = os.path.join(dqmc_folder + "/ejpi/", name_plaq)

    name_tau = f"l6b1js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
    # name_tau = f"l6b24js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
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
          

    # save plot
    method_name = "Splaq"
    save_dir = os.path.join(script_path, f"./figures/energies_Splaq")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")




if __name__ == '__main__':
    # Lx, Ly, Ltau = 6, 6, 10
    # Vs = Lx * Ly * Ltau

    plot_energy_J(starts=[2000], sample_steps=[1])

    dbstop = 1


