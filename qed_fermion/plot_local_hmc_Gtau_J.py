import json
import math
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
sys.path.insert(0, script_path + '/../')

from qed_fermion.hmc_sampler2_batch_fermion import HmcSampler
from qed_fermion.hmc_sampler2_batch_local import LocalUpdateSampler


def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), hmc_filename='', dqmc_filename='', local_update_filename='', specifics = '', starts=[500], ends=[1000001], sample_steps=[1], scale_it=[False]):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize


    idx_ref = 5

    # Parse specifics
    parts = specifics.split('_')
    jtau_index = parts.index('Jtau')  # Find position of 'Jtau'
    jtau_value = float(parts[jtau_index + 1])   # Get the next element
    

    # ======== Plot ======== #
    plt.figure()
    if len(hmc_filename):
        start = starts.pop(0)
        end = ends.pop(0)
        sample_step = sample_steps.pop(0)
        seq_idx = np.arange(start, end, sample_step)

        res = torch.load(hmc_filename)
        print(f'Loaded: {hmc_filename}')

        G_list = res['G_list']
        step = res['step']
        x = np.array(list(range(G_list[0].size(-1))))

        G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))
    
        plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_hmc', color='blue', lw=2)


    if len(dqmc_filename):   
        # name = f"l4b3js{jtau_value:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        data = np.genfromtxt(dqmc_filename)
        G_dqmc = np.concat([data[:, 0], data[:1, 0]])
        G_dqmc_err = np.concat([data[:, 1], data[:1, 1]])
        
        if scale_it.pop(0):
            ## Scale G_mean_anal to match G_mean based on their first elements
            scale_factor = G_mean[idx_ref] / G_dqmc[idx_ref]
            G_dqmc *= scale_factor
        
        x_dqmc = np.array(list(range(G_dqmc.size)))
        plt.errorbar(x_dqmc, G_dqmc, yerr=G_dqmc_err, linestyle='--', marker='*', label='G_dqmc', color='red', lw=2, ms=10)


    if len(local_update_filename): 
        res = torch.load(local_update_filename)
        print(f'Loaded: {local_update_filename}')

        G_list_local = res['G_list']
        step_local = res['step']
        x_local = np.array(list(range(G_list_local[0].size(-1))))

        start_local = starts.pop(0)
        end_local = ends.pop(0)
        sample_step_local = sample_steps.pop(0)
        seq_idx_local = np.arange(start_local, end_local, sample_step_local)

        G_local_mean = G_list_local[seq_idx_local].numpy().mean(axis=(0, 1))
        G_local_std = G_list_local[seq_idx_local].numpy().std(axis=(0, 1))

        if scale_it.pop(0):
            # scale_factor = G_dqmc[idx_ref] / G_local_mean[idx_ref] if len(dqmc_filename) else 1
            scale_factor = G_mean[idx_ref] / G_local_mean[idx_ref] if len(hmc_filename) else 1
            G_local_mean *= scale_factor
        
        plt.errorbar(x_local, G_local_mean, yerr=G_local_std/np.sqrt(seq_idx_local.size), linestyle='-', marker='o', label='G_local', color='green', lw=2)


    # Add labels and title
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$")
    plt.title(f"Ntau={Ltau} Nx=Ny={Lx} J={jtau_value} Nswp={end - start}")
    plt.legend()

    # Save plot
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens"
    save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


    # ======== Log plot ======== #
    plt.figure()
    if len(hmc_filename):
        plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_hmc', color='blue', lw=2)

    if len(dqmc_filename):
        plt.errorbar(x+1, G_dqmc, yerr=G_dqmc_err, linestyle='--', marker='*', label='G_dqmc', color='red', lw=2, ms=10)

    if len(local_update_filename):
        plt.errorbar(x_local+1, G_local_mean, yerr=G_local_std/np.sqrt(seq_idx_local.size), linestyle='', marker='o', label='G_local', color='green', lw=2)


    # Add labels and title
    plt.xlabel('X-axis label')
    plt.ylabel('log10(G) values')
    plt.title(f"Ntau={Ltau} Nx=Ny={Lx} J={jtau_value} Nswp={end - start}")
    plt.legend()
  
    plt.xscale('log')
    plt.yscale('log')


    # --------- save_plot ---------
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens_log"
    save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")



if __name__ == '__main__':
    # J = float(os.getenv("J"))
    # Nstep = int(os.getenv("Nstep"))

    Js = [0.5]

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
        # hmc
        step = Nstep
        hmc_filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{hmc.get_specifics()}_step_{step}.pt"
        
        # local
        step_lmc = lmc.N_step
        local_update_filename = script_path + f"/check_points/local_check_point/ckpt_N_{lmc.get_specifics()}_step_{step_lmc}.pt"

        # Measure
        Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
        load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc_filename, '', local_update_filename, specifics=hmc.get_specifics(), starts=[500, 200*lmc.Vs], ends=[Nstep, step_lmc], sample_steps=[1, 10], scale_it=[False, False])

        plt.show(block=False)

    dbstop = 1


