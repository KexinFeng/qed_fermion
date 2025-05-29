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

if __name__ == '__main__':
    # Create configs file
    Lx = int(os.getenv("Lx", '8'))
    print(f"Lx: {Lx}")
    Ltau = int(os.getenv("Ltau", '80'))
    print(f"Ltau: {Ltau}")
    # Ltau = Lx * 10

    Js = [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
    # Js = [0.5, 1.0, 3.0]

    bs = 2

    input_folder = "/Users/kx/Desktop/hmc/fignote/equilibrium_issue/hmc_check_point_bench/"

    end = 10000

    @time_execution
    def iterate_func():
        for J in Js:

            bid = 1

            hmc_filename = f"/stream_ckpt_N_t_hmc_{Lx}_Ltau_{Ltau}_Nstp_{end}_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_step_{end}.pt"

            # Load and write
            # [seq, Ltau * Ly * Lx * 2]
            boson_seq = torch.load(hmc_filename)
            # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
            print(f'Loaded: {hmc_filename}')  
            
                  

            dbstop = 1
            
  
    iterate_func()
