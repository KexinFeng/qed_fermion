import matplotlib.pyplot as plt

plt.ion()

import numpy as np

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../../')

from load_write2file_convert import time_execution
bs = 5
data_list = []

for bid in range(bs):
    file_path = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run_benchmark/run_meas_J_0.5_L_6_Ltau_10_bid{bid}_part_0_psz_500_start_5999_end_6000/spsm.bin"
    # Assuming torch.load is appropriate for loading the file
    data = np.genfromtxt(file_path)
    
    txt_file_path = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/post_processors/fermi_bench/Nrv_200/spsm_k_b{bid}.txt"
    data2 = np.genfromtxt(txt_file_path)

    # Compare the third column of data and data2 using torch.testing.assert_close
    torch.testing.assert_close(
        torch.tensor(data[:, 2]), 
        torch.tensor(data2[:, 2]), 
        atol=3e-2, 
        rtol=0
    )
    print(f"Assertion passed for batch id {bid}")
    max_abs_diff = np.max(np.abs(data[:, 2] - data2[:, 2]))
    print(f"Max abs diff for batch id {bid}: {max_abs_diff}")
    max_rel_diff = np.max(np.abs((data[:, 2] - data2[:, 2]) / data2[:, 2]))
    print(f"Max rel diff for batch id {bid}: {max_rel_diff}")





