import glob
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
import time
import subprocess
from tqdm import tqdm

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds\n")
        return result
    return wrapper

@time_execution
def load_write2file2(output_folder, Lsize=(6, 6, 10), hmc_filename='', starts=[500], sample_steps=[1], ends=[6000]):
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
    
    # Load and write
    # [seq, Ltau * Ly * Lx * 2]
    boson_seq = torch.load(hmc_filename)
    # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
    print(f'Loaded: {hmc_filename}')        
    
    # Extract Nstep and Nstep_local from filenames
    # hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    # end = int(hmc_match.group(1))
    end = ends.pop(0)

    start = starts.pop(0)
    sample_step = sample_steps.pop(0)
    seq_idx = set(list(range(start, end, sample_step)))

    # Write result to file
    output_file = "confin_all_confs"
    output_path = os.path.join(output_folder, output_file)
    with open(output_path, 'w') as f:
        filtered_seq = [(i, boson) for i, boson in enumerate(boson_seq) if i in seq_idx]
        for i, boson in tqdm(filtered_seq, desc="Processing boson sequences"):
            boson = convert((Lx, Ly, Ltau), boson).cpu().numpy()
            f.write("           1                     0  1.667169721062853E-002\n") 
            np.savetxt(f, boson, fmt="%.16E")

    print(f"Results saved to {output_path}")

    # Create or overwrite the ftdqmc.in file
    ftdqmc_file_path = os.path.join(output_folder, "ftdqmc.in")
    with open(ftdqmc_file_path, 'w') as f:
        f.write(f"L : {Lx} # <-\n")
        f.write("Nflavor : 2\n")
        f.write("rt : 0.0\n")
        f.write("init_xmag : 0\n")
        f.write("mu : 0.0\n")
        f.write("muA : 0.0\n")
        f.write("muB : 0.0\n")
        f.write("rj : 1.0\n")
        f.write("rhub : 0.0\n")
        f.write("jpi : 1.0\n")
        f.write(f"beta : {Ltau * 0.1:.1f} #  Ntau * dtau, 12.0 for confin_all_confs_demo # <-\n")
        f.write("dtau : 0.1\n")
        f.write(f"js : {jtau_value:.1f}\n")
        f.write("phi_box : 3.141592654\n")
        f.write("llocal : T\n")
        f.write("nsw_stglobal : 0\n")
        f.write("nsw_stglobal_slice : 0\n")
        f.write("ncumulate_bare : 0\n")
        f.write("nsweep : 1\n")
        f.write(f"nbin : {len(seq_idx)} # count of config samples, 10 # <-\n")
        f.write("lsstau : T\n")
        f.write("ltau : F\n")
        f.write("nwrap : 10\n")
        f.write("nuse : 0\n")
        f.write("rstep0 : 0.01\n")

    print(f"ftdqmc.in file created at {ftdqmc_file_path}")


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
    boson = boson.view(2, Lx, Ly, Ltau)
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

def clear(directory):
    # Clear previous builds (equivalent to `clear.sh`)
    if os.path.exists(directory):
        for pattern in ["*.bin", "*out", "klist"]:
            for filepath in glob.glob(os.path.join(directory, pattern)):
                os.remove(filepath)
        print(f"Cleared files in {directory}")

@time_execution
def execute_bash_scripts(directory):
    """
    Implements the functionality of the bash scripts `run_all_J.sh` and `clear.sh` in Python.
    """
    # Build the project (equivalent to `make`)
    src_folder = os.path.join(directory, "../../src")
    if os.path.exists(src_folder):
        subprocess.run(["make"], cwd=src_folder, check=True)
        print("Project built successfully.")

    # Execute the commands in each directory
    if os.path.exists(directory):
        ftdqmc_executable = os.path.join(directory, "../../src/ftdqmc")
        if os.path.exists(ftdqmc_executable):
            subprocess.run([ftdqmc_executable], cwd=directory, check=True)
            print(f"Executed ftdqmc in {directory}")
        else:
            print(f"Executable not found: {ftdqmc_executable}")
    else:
        print(f"Directory not found: {directory}")


if __name__ == '__main__':
    # Create configs file
    Lx = 6
    Ltau = Lx * 40
    Ltau = 10
    # Js = [1.0, 1.5, 2.0, 2.5, 3.0]
    Js = [0.5, 1.0, 3.0]
    input_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/benchmark_6x6x10/ckpt/hmc_check_point_unconverted_stream"
    # input_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/hmc_check_point_L6"
    for J in Js:
        output_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run2/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}/"
        os.makedirs(output_folder, exist_ok=True)

        hmc_filename = f"/stream_ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_6000_bs1_Jtau_{J:.2g}_K_1_dtau_0.1_step_6000.pt"
        load_write2file2(output_folder, Lsize=(Lx, Lx, Ltau), hmc_filename=input_folder + hmc_filename, starts=[2000], sample_steps=[5], ends=[6000])
    
    # Run
    for J in Js:
        output_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run2/run_meas_J_{J:.2g}_L_{Lx}_Ltau_{Ltau}/"
        clear(output_folder)
        execute_bash_scripts(output_folder)

