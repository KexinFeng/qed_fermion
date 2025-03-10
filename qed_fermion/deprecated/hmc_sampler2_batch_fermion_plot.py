import json
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



def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), step=1000001, specifics='', plot_anal=True, plot_dqmc=True, start=500, sample_step=1):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    if len(specifics) == 0:
        filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    else:
        filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{specifics}_step_{step}.pt"
    res = torch.load(filename)
    print(f'Loaded: {filename}')

    G_list = res['G_list']
    step = res['step']
    x = np.array(list(range(G_list[0].size(-1))))

    # start = 2000
    end = step
    # sample_step = 1
    seq_idx = np.arange(start, end, sample_step)

    # Parse specifics
    parts = specifics.split('_')
    jtau_index = parts.index('Jtau')  # Find position of 'Jtau'
    jtau_value = float(parts[jtau_index + 1])   # Get the next element
    
    # ======== Plot ======== #
    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))


    if plot_dqmc:
        # Analytical data
        data_folder_anal = script_path + "/../../benchmark_dqmc/piflux_B0.0K1.0_tuneJ_kexin_hk/photon_mass_sin_splaq/"

        name = f"l4b3js{jtau_value:.1f}jpi1.0mu0.0nf2_dqmc_bin.dat"

        pkl_filepath = os.path.join(data_folder_anal, name)
        data = np.genfromtxt(pkl_filepath)
        G_dqmc = np.concat([data[:, 0], data[:1, 0]])
        G_dqmc_err = np.concat([data[:, 1], data[:1, 1]])
        
        # Scale G_mean_anal to match G_mean based on their first elements
        idx_ref = 1
        scale_factor = G_mean[idx_ref] / G_dqmc[idx_ref]
        # G_dqmc *= scale_factor

        plt.errorbar(x, G_dqmc, yerr=G_dqmc_err, linestyle='--', marker='*', label='G_dqmc', color='red', lw=2, ms=10)

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
    plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))

    if plot_anal:
        # Analytical data
        data_folder_anal = script_path + "/../../hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
        name = f"N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{Ltau}_num_{Ltau}"
        pkl_filepath = os.path.join(data_folder_anal, f"{name}.pkl")

        with open(pkl_filepath, 'r') as f:
            loaded_data_anal = json.load(f)

        G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1]

        # Scale G_mean_anal to match G_mean based on their first elements
        idx_ref = 1
        scale_factor_anal = G_mean[idx_ref] / G_mean_anal[idx_ref]
        G_mean_anal *= scale_factor_anal

        plt.plot(x+1, G_mean_anal, linestyle='--', marker='*', label='G_avg_anal', color='red', lw=2, ms=10)

    if plot_dqmc:
        plt.errorbar(x+1, G_dqmc, yerr=G_dqmc_err, linestyle='--', marker='*', label='G_dqmc', color='red', lw=2, ms=10)


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
    J = float(os.getenv("J"))
    Nstep = int(os.getenv("Nstep"))

    Js = [0.5, 1, 3]
    Js = [0.5]
    for J in Js:
        Nstep = 8000
        print(f'J={J} \nNstep={Nstep}')

        hmc = HmcSampler(J=J, Nstep=Nstep)
        hmc.reset()
        
        step = 8000
        # Measure
        Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
        load_visualize_final_greens_loglog((Lx, Ly, Ltau), step=step, specifics=hmc.specifics, plot_anal=False, plot_dqmc=True, start=2000, sample_step=1)

        plt.show(block=True)

    dbstop = 1


