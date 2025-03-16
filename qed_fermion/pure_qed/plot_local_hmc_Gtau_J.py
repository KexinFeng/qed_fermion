import json
import math
import re
import matplotlib.pyplot as plt
plt.ion()

import numpy as np
# matplotlib.use('MacOSX')
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../')

from qed_fermion.hmc_sampler_batch import HmcSampler
from qed_fermion.local_sampler_batch import LocalUpdateSampler


def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), hmc_filename='', local_update_filename='', specifics = '', starts=[500], sample_steps=[1], scale_it=[False]):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    idx_ref = 5

    # Parse to get specifics
    path_parts = hmc_filename.split('/')
    filename = path_parts[-1]
    filename_parts = filename.split('_')
    specifics = '_'.join(filename_parts[1:]).replace('.pt', '')
    print(f"Parsed specifics: {specifics}")

    # Parse specifics
    parts = hmc_filename.split('_')
    jtau_index = parts.index('Jtau')  # Find position of 'Jtau'
    jtau_value = float(parts[jtau_index + 1])   # Get the next element

    # ======== Plot ======== #
    plt.figure()
    if len(hmc_filename):
        res = torch.load(hmc_filename)
        print(f'Loaded: {hmc_filename}')        
        
        # Extract Nstep and Nstep_local from filenames
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))

        start = starts.pop(0)
        sample_step = sample_steps.pop(0)
        seq_idx = np.arange(start, end, sample_step)


        G_list = res['G_list']
        x = np.array(list(range(G_list[0].size(-1))))

        G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))
    
        plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_hmc', color='blue', lw=2)

        for idx, bi in enumerate(range(G_list.size(1))):
            plt.errorbar(
                x, 
                G_list[seq_idx, bi].numpy().mean(axis=(0)), 
                yerr=G_list[seq_idx, bi].numpy().std(axis=(0))/np.sqrt(seq_idx.size),
                alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='o', lw=2, color=f"C{idx}")


    if len(local_update_filename): 
        res = torch.load(local_update_filename)
        print(f'Loaded: {local_update_filename}')

        # Extract Nstep and Nstep_local from filenames
        local_match = re.search(r'Nstp_(\d+)', local_update_filename)
        end_local = int(local_match.group(1))
        start_local = starts.pop(0)
        sample_step_local = sample_steps.pop(0)
        seq_idx_local = np.arange(start_local, end_local, sample_step_local)

        G_list_local = res['G_list']
        x_local = np.array(list(range(G_list_local[0].size(-1))))

        batch_idx = torch.tensor([0, 1, 4])
        batch_size = G_list_local.size(1)
        batch_idx = torch.arange(batch_size)
        
        G_local_mean = G_list_local[seq_idx_local][:, batch_idx].numpy().mean(axis=(0, 1))
        G_local_std = G_list_local[seq_idx_local][:, batch_idx].numpy().std(axis=(0, 1))

        if scale_it.pop(0):
            scale_factor = G_mean[idx_ref] / G_local_mean[idx_ref] if len(hmc_filename) else 1
            G_local_mean *= scale_factor
        
        plt.errorbar(x_local, G_local_mean, yerr=G_local_std/np.sqrt(seq_idx_local.size), linestyle='-', marker='*', markersize=10, label=f'G_local_{batch_idx.tolist()}', color='green', lw=2)

        for idx, bi in enumerate(range(G_list.size(1))):
            plt.errorbar(
            x_local, 
            G_list_local[seq_idx_local, bi].numpy().mean(axis=(0)), 
            yerr=G_list_local[seq_idx_local, bi].numpy().std(axis=(0))/np.sqrt(seq_idx_local.size),
            alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='*', markersize=10, lw=2, color=f"C{idx}")


    # Add labels and title
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$")
    plt.title(f"Ntau={Ltau} Nx=Ny={Lx} J={jtau_value} Nswp={end - start}")
    plt.legend(ncol=2)

    # Save plot
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens"
    save_dir = os.path.join(script_path, f"./figures/{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    # ======== Log plot ======== #
    plt.figure()
    if len(hmc_filename):
        plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_hmc', color='blue', lw=2)

    if len(local_update_filename):
        plt.errorbar(x_local+1, G_local_mean, yerr=G_local_std/np.sqrt(seq_idx_local.size), linestyle='', marker='*', markersize=10, label='G_local', color='green', lw=2)

    # Add labels and title
    plt.xlabel('X-axis label')
    plt.ylabel('log10(G) values')
    plt.title(f"Ntau={Ltau} Nx=Ny={Lx} J={jtau_value} Nswp={end - start}")
    plt.legend(ncol=2)
  
    plt.xscale('log')
    plt.yscale('log')

    # --------- save_plot ---------
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens_log"
    save_dir = os.path.join(script_path, f"./figures/{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

if __name__ == '__main__':
    Js = [0.5, 1, 3]

    for J in Js:
        Nstep = 5000

        hmc = HmcSampler(J=J, Nstep=Nstep)
        hmc.Lx, hmc.Ly, hmc.Ltau = 6, 6, 10
        hmc.reset()

        Nstep_local = 1000
        lmc = LocalUpdateSampler(J=J, Nstep=Nstep_local)
        lmc.Lx, lmc.Ly, lmc.Ltau = 6, 6, 10
        lmc.reset()

        # File names
        # # hmc
        # step = Nstep
        # hmc_filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{hmc.get_specifics()}_step_{step}.pt"
        
        # # local
        # step_lmc = lmc.N_step
        # local_update_filename = script_path + f"/check_points/local_check_point/ckpt_N_{lmc.get_specifics()}_step_{step_lmc}.pt"

        # hmc_filename = script_path + "/check_points/hmc_check_point/ckpt_N_hmc_6_Ltau_10_Nstp_10000_Jtau_0.5_K_1_dtau_0.1_step_10000.pt"
        # local_update_filename = script_path + "/check_points/local_check_point/ckpt_N_local_6_Ltau_10_Nstp_3600000_Jtau_0.5_K_1_dtau_0.1_step_3600000.pt"

        hmc_filename = f"/Users/kx/Desktop/hmc/fignote/local_vs_hmc_check_fermion/hmc_check_point/ckpt_N_hmc_6_Ltau_10_Nstp_10000_bs5_Jtau_{J:.1g}_K_1_dtau_0.1_step_10000.pt"
        local_update_filename = f"/Users/kx/Desktop/hmc/fignote/local_vs_hmc_check_fermion/local_check_point/ckpt_N_local_6_Ltau_10_Nstp_3600000bs_5_Jtau_{J:.1g}_K_1_dtau_0.1_step_3600000.pt"

        # Measure
        Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
        load_visualize_final_greens_loglog(
            (Lx, Ly, Ltau), 
            hmc_filename, local_update_filename, 
            specifics=hmc.get_specifics(), 
            starts=[3000, 3000*lmc.Vs], 
            sample_steps=[1, lmc.Vs], 
            scale_it=[False, False])

    plt.show(block=True)

    dbstop = 1


