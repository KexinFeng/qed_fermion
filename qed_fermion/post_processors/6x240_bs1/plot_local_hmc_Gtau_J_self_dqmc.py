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
sys.path.insert(0, script_path + '/../../../')


from qed_fermion.utils.stat import t_based_error, std_root_n, error_mean
import time

# Add partition parameters
part_size = 500
start_dqmc = 2000
end_dqmc = 6000

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
        res = torch.load(hmc_filename, map_location='cpu')
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
        # Extract J and L values from the dqmc_filename
        dqmc_dir = os.path.dirname(dqmc_filename)
        match = re.search(r'J_([\d\.]+)_L_(\d+)', dqmc_dir)
        
        if match:
            J = float(match.group(1))
            L = int(match.group(2))
            
            # Aggregate DQMC data from all parts
            all_data = []
            
            num_parts = math.ceil((end_dqmc - start_dqmc) / part_size)
            for part_id in range(num_parts):
                part_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run3/run_meas_J_{J:.2g}_L_{L}_Ltau_{Ltau}_part_{part_id}_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"
                name = os.path.basename(dqmc_filename)
                part_filename = os.path.join(part_folder, name)
                
                try:
                    part_data = np.genfromtxt(part_filename).reshape(-1, Ltau)
                    all_data.append(part_data)
                    print(f'Loaded DQMC data: {part_filename}')
                except (FileNotFoundError, ValueError) as e:
                    raise RuntimeError(f'Warning: Error loading {part_filename}: {str(e)}') from e
            
            # Concatenate data from all parts
            data = np.concatenate(all_data)
            print(f"Combined data from {len(all_data)} parts, total samples: {data.shape[0]}")
        else:
            # If no match, use the original file
            raise RuntimeError(f'No match found: {dqmc_filename}')
        
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
    Js = [1.0, 1.5, 2.0, 2.5, 3.0]
    # Js = [0.5, 1.0, 3.0]

    for J in Js:
        hmc_folder = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_unconverted_stream/"
        hmc_file = f"ckpt_N_hmc_6_Ltau_10_Nstp_6000_bs1_Jtau_{J:.1g}_K_1_dtau_0.1_step_6000.pt"
        hmc_folder = f"/Users/kx/Desktop/hmc/fignote/ftdqmc/hmc_check_point_L6x240"
        hmc_file = f"ckpt_N_hmc_6_Ltau_240_Nstp_6000_bs1_Jtau_{J:.2g}_K_1_dtau_0.1_step_6000.pt"
       
        hmc_filename = os.path.join(hmc_folder, hmc_file)

        # Update to use the run3 folder structure with partitioned files
        # dqmc_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run2/run_meas_J_{J:.2g}_L_6/"
        dqmc_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run3/run_meas_J_{J:.2g}_L_6_Ltau_240_part_0_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"
        name = f"thetacorrtau_sin_splaq.bin"
        dqmc_filename = os.path.join(dqmc_folder, name)

        # Measure
        Lx, Ly, Ltau = 6, 6, 240
        # Lx, Ly, Ltau = 6, 6, 10

        load_visualize_final_greens_loglog(
            (Lx, Ly, Ltau), 
            hmc_filename, dqmc_filename, 
            starts=[2000], 
            sample_steps=[1])

    plt.show(block=True)

    dbstop = 1


