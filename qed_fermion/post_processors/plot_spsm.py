import re
import time
import matplotlib.pyplot as plt

plt.ion()

import numpy as np
import math

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator

from load_write2file_convert import time_execution

# Add partition parameters
part_size = 500
start_dqmc = 2000
end_dqmc = 6000
root_folder = "/Users/kx/Desktop/forked/dqmc_u1sl_mag/run3/"

@time_execution
def plot_spsm(Lsize=(6, 6, 10)):
    Js = [1.0, 1.5, 2.0, 2.5, 3.0]
    r_afm_values = []
    r_afm_errors = []
    
    Lx, Ly, Ltau = Lsize
    vs = Lx**2
    
    plt.figure(figsize=(8, 6))
    
    for J in Js:
        # Initialize data collection arrays for this J value
        all_data = []
        
        # Calculate the number of parts
        num_parts = math.ceil((end_dqmc - start_dqmc) / part_size)
        
        # Loop through all parts
        for part_id in range(num_parts):
            input_folder = root_folder + f"/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}_part_{part_id}_psz_{part_size}_start_{start_dqmc}_end_{end_dqmc}/"
            name = f"spsm.bin"
            dqmc_filename = os.path.join(input_folder, name)
            
            try:
                part_data = np.genfromtxt(dqmc_filename)
                all_data.append(part_data)
                print(f'Loaded DQMC data: {dqmc_filename}')
            except (FileNotFoundError, ValueError) as e:
                raise RuntimeError(f'Error loading {dqmc_filename}: {str(e)}') from e
        
        # Combine all parts' data
        data = np.concatenate(all_data)
        data = data.reshape(-1, vs, 4)
        # data has shape [num_sample, vs, 4], where the last dim has entries: kx, ky, val, error. 
        # [num_sample]
        r_afm = 1 - data[:, 1, 2] / data[:, 0, 2]
        rtol = data[:, :, 3] / data[:, :, 2]
        r_afm_err = abs(rtol[:, 0] - rtol[:, 1]) * (1 - r_afm)
        
        # Calculate mean and error for plotting
        r_afm_mean = np.mean(r_afm)
        r_afm_error = np.mean(r_afm_err)
        
        r_afm_values.append(r_afm_mean)
        r_afm_errors.append(r_afm_error)

    # Plot the errorbar for the means
    plt.errorbar(Js, r_afm_values, yerr=r_afm_errors, 
                linestyle='-', marker='o', lw=2, color='blue', label='r_afm')
    
    plt.xlabel('J', fontsize=14)
    plt.ylabel('r_afm', fontsize=14)
    plt.title(f'r_afm vs J (L={Lx})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # save plot
    method_name = "spsm"
    save_dir = os.path.join(script_path, f"./figures/r_afm")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_L{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")
    
    return r_afm_values, r_afm_errors



if __name__ == '__main__':
    Lx = 6
    Ltau = Lx*40
    plot_spsm(Lsize=(Lx, Lx, Ltau))
    plt.show(block=True)



