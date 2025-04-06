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
def load_write2file2(Lsize=(6, 6, 10), hmc_filename='', starts=[500], sample_steps=[1]):
    Lx, Ly, Ltau = Lsize

    # Parse to get specifics
    path_parts = hmc_filename.split('/')
    filename = path_parts[-1]
    filename_parts = filename.split('_')
    specifics = '_'.join(filename_parts[1:]).replace('.pt', '')
    print(f"Parsed specifics: {specifics}")

    parts = hmc_filename.split('_')
    jtau_index = parts.index('Jtau')  # Find position of 'Jtau'
    jtau_value = float(parts[jtau_index + 1])   # Get the next element
    
    if len(hmc_filename):
        # [seq, Ltau * Ly * Lx * 2]
        boson_seq = torch.load(hmc_filename)
        # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
        print(f'Loaded: {hmc_filename}')        
        
        # Extract Nstep and Nstep_local from filenames
        hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
        end = int(hmc_match.group(1))

        start = starts.pop(0)
        sample_step = sample_steps.pop(0)
        seq_idx = set(list(range(start, end, sample_step)))

        # Write result to file
        # output_file_name = f'confin_all_confs_J_{jtau_value:.1g}_Lx_{Lx}_Ltau_{Ltau}'
        # output_path = os.path.join(script_path + '/check_points/hmc_check_point/', output_file_name)

        output_path = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run/run_meas_J_{jtau_value:.1g}/confin_all_confs"
        with open(output_path, 'w') as f:
            for i, boson in enumerate(boson_seq):
                if i not in seq_idx: continue
                boson = convert((Lx, Ly, Ltau), boson).cpu().numpy()
                f.write("           1                     0  1.667169721062853E-002\n") 
                np.savetxt(f, boson, fmt="%.16E")

        print(f"Results saved to {output_path}")

def convert(sizes, boson):
    # Transpose indices
    Lx, Ly, Ltau = sizes
    device = boson.device

    xs = torch.arange(0, Lx, 2, device=device, dtype=torch.int64).unsqueeze(0)
    ys = torch.arange(0, Ly, 2, device=device, dtype=torch.int64).unsqueeze(1)
    d_i_list_1 = (xs + ys * Lx).view(-1)
    d_i_list_3 = ((xs - 1)%Lx + ys * Lx).view(-1)
    d_i_list_4 = (xs + (ys-1)%Ly * Lx).view(-1)
    d_i_list_1 = d_i_list_1.view(Ly // 2, Lx // 2).T
    d_i_list_3 = d_i_list_3.view(Ly // 2, Lx // 2).T
    d_i_list_4 = d_i_list_4.view(Ly // 2, Lx // 2).T

    xs2 = torch.arange(1, Lx, 2, device=device, dtype=torch.int64).unsqueeze(0)
    ys2 = torch.arange(1, Ly, 2, device=device, dtype=torch.int64).unsqueeze(1)
    dd_i_list_1 = (xs2 + ys2 * Lx).view(-1)
    dd_i_list_3 = ((xs2 - 1) % Lx + ys2 * Lx).view(-1)
    dd_i_list_4 = (xs2 + (ys2-1)%Ly * Lx).view(-1)
    dd_i_list_1 = dd_i_list_1.view(Ly // 2, Lx // 2).T
    dd_i_list_3 = dd_i_list_3.view(Ly // 2, Lx // 2).T
    dd_i_list_4 = dd_i_list_4.view(Ly // 2, Lx // 2).T
    
    intertwined_i_list_1 = torch.stack((d_i_list_1, dd_i_list_1), dim=0).transpose(0, 1).flatten()
    intertwined_i_list_3 = torch.stack((d_i_list_3, dd_i_list_3), dim=0).transpose(0, 1).flatten()
    intertwined_i_list_4 = torch.stack((d_i_list_4, dd_i_list_4), dim=0).transpose(0, 1).flatten()

    # [2, Lx, Ly, Ltau]
    result = []
    for tau in range(Ltau):
        # [2, Lx, Ly]
        boson_x = boson[0, :, :, tau].T.flatten()
        boson_y = boson[1, :, :, tau].T.flatten()
        
        result.append(boson_x[intertwined_i_list_1])
        result.append(boson_y[intertwined_i_list_1])
        result.append(-boson_x[intertwined_i_list_3])
        result.append(-boson_y[intertwined_i_list_4])
        
    return torch.cat(result, dim=0).view(-1)


if __name__ == '__main__':
    Lx = 6
    Ltau = 10
    Js = [0.5,1,3]
    for J in Js:
        hmc_filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_unconverted/stream_ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_6000_bs1_Jtau_{J:.1g}_K_1_dtau_0.1_step_6000.pt"
        load_write2file2(Lsize=(Lx, Lx, Ltau), hmc_filename=hmc_filename, starts=[2000], sample_steps=[1])

