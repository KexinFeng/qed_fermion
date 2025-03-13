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


def load_energy_perJ(Lsize=(20, 20, 20), hmc_filename='', dqmc_filename='', local_update_filename='', specifics = '', starts=[500], ends=[1000001], sample_steps=[1], scale_it=[False]):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    Sbs = []
    Sfs = []

    # ======== Plot ======== #
    plt.figure()
    if len(hmc_filename):
        start = starts.pop(0)
        end = ends.pop(0)
        sample_step = sample_steps.pop(0)
        seq_idx = np.arange(start, end, sample_step)

        res = torch.load(hmc_filename)
        print(f'Loaded: {hmc_filename}')

        Sb_list = res['S_plaq_list'][seq_idx]
        Sf_list = res['Sf_list'][seq_idx]
        # Sf_list = res['Sf_list'][seq_idx]

        Sbs.append((Sb_list.mean(dim=0), Sb_list.std(dim=0)/math.sqrt(len(Sb_list))))
        Sfs.append((Sf_list.mean(dim=0), Sf_list.std(dim=0)/math.sqrt(len(Sf_list))))

    if len(dqmc_filename):   
        # name = f"l4b3js{jtau_value:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        data = np.genfromtxt(dqmc_filename)
        S_plaq_list = data

        Sbs.append((S_plaq_list.mean(), S_plaq_list.std()/math.sqrt(len(S_plaq_list))))
        Sfs.append((torch.tensor([0]), torch.tensor([0])))

    if len(local_update_filename): 
        res = torch.load(local_update_filename)
        print(f'Loaded: {local_update_filename}')

        start_local = starts.pop(0)
        end_local = ends.pop(0)
        sample_step_local = sample_steps.pop(0)
        seq_idx_local = np.arange(start_local, end_local, sample_step_local)

        Sb_list = res['S_plaq_list'][seq_idx_local]
        Sf_list = res['Sf_list'][seq_idx_local]

        Sbs.append((Sb_list.mean(dim=0), Sb_list.std(dim=0)/math.sqrt(len(Sb_list))))
        Sfs.append((Sf_list.mean(dim=0), Sf_list.std(dim=0)/math.sqrt(len(Sf_list))))

    return Sbs, Sfs


def plot_energy_J(Js, Nstep=3000, Nstep_local=100):

    boson_lines = [list() for _ in range(3)]
    boson_err_lines = [list() for _ in range(3)]
    fermion_lines = [list() for _ in range(3)]
    fermion_err_lines = [list() for _ in range(3)]
    legends = ['hmc', 'local']
    xs = Js

    for J in Js:
        print(f'J={J} \nNstep={Nstep}')

        hmc = HmcSampler(J=J, Nstep=Nstep)
        hmc.Lx, hmc.Ly, hmc.Ltau = 6, 6, 10
        # hmc.delta_t = 0.02
        hmc.reset()

        lmc = LocalUpdateSampler(J=J, Nstep=Nstep_local)
        lmc.Lx, lmc.Ly, lmc.Ltau = 6, 6, 10
        lmc.reset()
    
        # File names
        step = Nstep
        hmc_filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{hmc.specifics}_step_{step}.pt"

        dqmc_folder = script_path + "/../../benchmark_dqmc/piflux_B0.0K1.0_L6_tuneJ_kexin_hk/ejpi/"
        name = f"l6b1js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        dqmc_filename = os.path.join(dqmc_folder, name)

        step_lmc = 36000
        local_update_filename = script_path + f"/check_points/local_check_point/ckpt_N_{lmc.specifics}_step_{step_lmc}.pt"

        # Load
        Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
        Sbs, Sfs = load_energy_perJ((Lx, Ly, Ltau), hmc_filename, '', local_update_filename, specifics=hmc.specifics, starts=[1000, 20000], ends=[Nstep, step_lmc], sample_steps=[1, 50])

        # Collect
        for Sb, ys, yerrs in zip(Sbs, boson_lines, boson_err_lines):
            ys.append(Sb[0].item())
            yerrs.append(Sb[1].item())

        for Sf, zs, zerrs in zip(Sfs, fermion_lines, fermion_err_lines):
            zs.append(Sf[0].item())
            zerrs.append(Sf[1].item())

    # Plot Sb
    plt.figure()
    for idx, label in enumerate(legends):
        plt.errorbar(xs, boson_lines[idx], yerr=boson_err_lines[idx], linestyle='-', marker='o', label=legends[idx], lw=2)

    plt.xlabel(r"$J$")
    plt.ylabel(r"$S_{plaq}$")
    # plt.title(f"Boo")
    plt.legend()

    # save plot
    method_name = "boson"
    save_dir = os.path.join(script_path, f"./figures/energies_boson")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    # Plot Sf
    plt.figure()
    for idx, label in enumerate(legends):
        plt.errorbar(xs, fermion_lines[idx], yerr=fermion_err_lines[idx], linestyle='-', marker='o', label=legends[idx], lw=2)

    plt.xlabel(r"$J$")
    plt.ylabel(r"$-log(detM)$")
    # plt.title(f"Boo")
    plt.legend()

    # save plot
    method_name = "fermion"
    save_dir = os.path.join(script_path, f"./figures/energies_fermion")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    plt.show(block=False)


def plot_Gtau_J(Js):
    for J in Js:
        Nstep = 3000
        print(f'J={J} \nNstep={Nstep}')

        hmc = HmcSampler(J=J, Nstep=Nstep)
        hmc.Lx, hmc.Ly, hmc.Ltau = 6, 6, 10
        # hmc.delta_t = 0.02
        hmc.reset()

        lmc = LocalUpdateSampler(J=J, Nstep=1e2)
        lmc.Lx, lmc.Ly, lmc.Ltau = 6, 6, 10
        lmc.reset()

        # File names
        step = Nstep
        hmc_filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{hmc.specifics}_step_{step}.pt"

        dqmc_folder = script_path + "/../../benchmark_dqmc/piflux_B0.0K1.0_L6_tuneJ_kexin_hk/photon_mass_sin_splaq/"
        name = f"l6b1js{J:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"
        dqmc_filename = os.path.join(dqmc_folder, name)

        step_lmc = 36000
        local_update_filename = script_path + f"/check_points/local_check_point/ckpt_N_{lmc.specifics}_step_{step_lmc}.pt"

        # Measure
        Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
        load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc_filename, dqmc_filename, local_update_filename, specifics=hmc.specifics, starts=[1000, 20000], ends=[Nstep, step_lmc], sample_steps=[1, 50], scale_it=[False, False])

        plt.show(block=False)


if __name__ == '__main__':
    # J = float(os.getenv("J"))
    # Nstep = int(os.getenv("Nstep"))

    Js = [0.5, 1, 3]
    Js = [0.5]

    plot_Gtau_J(Js)
    # plot_energy_J(Js)

    dbstop = 1


