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
    spsm_k_dqmc = np.genfromtxt(file_path)

    se_file_path = script_path + f"/data_se_start-1/Lx_6_Ltau_10_Nrv_100_mxitr_400/spsm_k_b{bid}.pt"
    data_se = torch.load(se_file_path, map_location='cpu', weights_only=False)
    spsm_k_se = data_se['mean'][0].reshape(-1).astype(np.float64)  # [bs, kx, ky]

    # Compare the third column of data and data_se using torch.testing.assert_close
    torch.testing.assert_close(
        torch.tensor(spsm_k_dqmc[:, 2]), 
        torch.tensor(spsm_k_se), 
        atol=3e-2, 
        rtol=0
    )
    print(f"Assertion passed for batch id {bid}")
    max_abs_diff = np.max(np.abs(spsm_k_dqmc[:, 2] - spsm_k_se))
    print(f"Max abs diff for batch id {bid}: {max_abs_diff:.3g}")
    max_rel_diff = np.max(np.abs((spsm_k_dqmc[:, 2] - spsm_k_se) / spsm_k_se))
    print(f"Max rel diff for batch id {bid}: {max_rel_diff:.3g}")





