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
from qed_fermion.utils.stat import t_based_error, std_root_n, init_convex_seq_estimator, error_mean
import time

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@time_execution
def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), hmc_filename='', dqmc_filename='', starts=[500], sample_steps=[1]):
    """
    Visualize green functions with error bar
    """
    # Load numerical data
    Lx, Ly, Ltau = Lsize

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
        seq_idx_init = np.arange(0, end, sample_step)


        G_list = res['G_list']
        x = np.array(list(range(G_list[0].size(-1))))

        G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))
    
        # err1 = error_mean(init_convex_seq_estimator(G_list[seq_idx_init].numpy())/ np.sqrt(seq_idx_init.size), axis=0) * 1
        err1 = error_mean(std_root_n(G_list[seq_idx].numpy(), axis=0, lag_sum=50), axis=0)
        err2 = t_based_error(G_list[seq_idx].mean(axis=0).numpy())
        # print(err1, '\n', err2)
        err_hmc = np.sqrt(err1**2)

        plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=err_hmc, linestyle='-', marker='o', label='G_hmc', color='blue', lw=2)

        for idx, bi in enumerate(range(G_list.size(1))):
            plt.errorbar(
                x, 
                G_list[seq_idx, bi].numpy().mean(axis=(0)), 
                # yerr=G_list[seq_idx, bi].numpy().std(axis=(0))/np.sqrt(seq_idx.size / 100),
                alpha=0.5, label=f'bs_{bi}', linestyle='--', marker='o', lw=2, color=f"C{idx}")

    if len(dqmc_filename):
        data = np.genfromtxt(dqmc_filename).reshape(-1, Ltau)
        # G_dqmc = np.concat([data[:, 0], data[:1, 0]])
        # G_dqmc_err = np.concat([data[:, 1], data[:1, 1]])
        G_dqmc = data.mean(axis=0)
        G_dqmc_err = data.std(axis=0) / np.sqrt(data.shape[0])
        
        x_dqmc = np.array(list(range(G_dqmc.size)))
        plt.errorbar(x_dqmc, G_dqmc, yerr=G_dqmc_err, linestyle='--', marker='*', label='G_dqmc', color='red', lw=2, ms=10)
        

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
        plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=err_hmc, linestyle='', marker='o', label='G_hmc', color='blue', lw=2)

    if len(dqmc_filename):
        plt.errorbar(x_dqmc + 1, G_dqmc, yerr=G_dqmc_err * 1, linestyle='--', marker='*', label='G_dqmc', color='red', lw=2, ms=10)

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
        Nstep = 10000

        hmc_folder = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_unconverted_stream/"
        hmc_file = f"ckpt_N_hmc_6_Ltau_10_Nstp_6000_bs1_Jtau_{J:.1g}_K_1_dtau_0.1_step_6000.pt"
        hmc_filename = os.path.join(hmc_folder, hmc_file)

        dqmc_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run/run_meas_J_{J:.1g}/"
        name = f"thetacorrtau_sin_splaq.bin"
        dqmc_filename = os.path.join(dqmc_folder, name)

        # Measure
        Lx, Ly, Ltau = 6, 6, 10
        load_visualize_final_greens_loglog(
            (Lx, Ly, Ltau), 
            hmc_filename, dqmc_filename, 
            starts=[2000], 
            sample_steps=[1])

    plt.show(block=True)

    dbstop = 1


