import torch
import os
import re
from pathlib import Path
import numpy as np

def parse_checkpoint_filename(filename):
    """
    Parse checkpoint filename to extract parameters.
    
    Example: ckpt_N_hmc_10_Ltau_100_Nstp_10000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_10000.pt
    """
    # Extract parameters using regex
    pattern = r'N_hmc_(\d+)_Ltau_(\d+)_Nstp_(\d+)_bs(\d+)_Jtau_([\d.]+)_K_(\d+)_dtau_([\d.]+)_delta_([\d.]+)_N_leapfrog_(\d+)_m_(\d+)_cg_rtol_([\d.e-]+)_max_iter_(\d+)_max_block_idx_(\d+)_gear(\d+)_steps_(\d+)_dt_deque_max_len_(\d+)_Nrv_(\d+)_cmp_(\w+)_step_(\d+)'
    
    match = re.search(pattern, filename)
    if match:
        params = {
            'N_hmc': int(match.group(1)),
            'Ltau': int(match.group(2)),
            'Nstp': int(match.group(3)),
            'bs': int(match.group(4)),
            'Jtau': float(match.group(5)),
            'K': int(match.group(6)),
            'dtau': float(match.group(7)),
            'delta': float(match.group(8)),
            'N_leapfrog': int(match.group(9)),
            'm': int(match.group(10)),
            'cg_rtol': float(match.group(11)),
            'max_iter': int(match.group(12)),
            'max_block_idx': int(match.group(13)),
            'gear': int(match.group(14)),
            'steps': int(match.group(15)),
            'dt_deque_max_len': int(match.group(16)),
            'Nrv': int(match.group(17)),
            'cmp': match.group(18) == 'True',
            'step': int(match.group(19))
        }
        return params
    else:
        print(f"Warning: Could not parse filename {filename}")
        return None

def load_checkpoint_data(file_path):
    """
    Load checkpoint data from a .pt file.
    """
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    # Define the checkpoint file paths
    checkpoint_files = [
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_10_Ltau_100_Nstp_8000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_8000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_12_Ltau_120_Nstp_8000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_8000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_16_Ltau_160_Nstp_8000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_8000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_20_Ltau_200_Nstp_8000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_8000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_26_Ltau_260_Nstp_8000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_8000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_30_Ltau_300_Nstp_8000_bs2_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_8000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_36_Ltau_360_Nstp_10000_bs1_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_10000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_40_Ltau_400_Nstp_10000_bs1_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_10000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_46_Ltau_460_Nstp_10000_bs1_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_10000.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_50_Ltau_500_Nstp_6500_bs1_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_40_cmp_False_step_6500.pt",
    "/Users/kx/Desktop/hmc/fignote/spsm_r_tau/ckpt_N_hmc_56_Ltau_560_Nstp_6500_bs1_Jtau_1.2_K_1_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_iter_400_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_Nrv_30_cmp_False_step_6500.pt",
    ]
    
    # Dictionary to store all loaded data
    loaded_data = {}
    
    print("Loading checkpoint files...")
    print("=" * 80)
    
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"  ERROR: File not found: {file_path}")
            continue
        
        # Parse parameters from filename
        params = parse_checkpoint_filename(filename)
        if params is None:
            continue
        
        # Load the checkpoint data
        data = load_checkpoint_data(file_path)
        if data is None:
            continue
        
        # Store data with key based on N_hmc and Ltau
        key = f"N{params['N_hmc']}_Ltau{params['Ltau']}"
        loaded_data[key] = {
            'params': params,
            'data': data,
            'file_path': file_path
        }
        
        print(f"  ✓ Successfully loaded")
        print(f"    Parameters: N_hmc={params['N_hmc']}, Ltau={params['Ltau']}, K={params['K']}")
        print(f"    Data type: {type(data)}")
        
        # Show data structure if it's a dictionary
        if isinstance(data, dict):
            print(f"    Data keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"      {k}: tensor shape {v.shape}, dtype {v.dtype}")
                elif isinstance(v, (list, tuple)):
                    print(f"      {k}: {type(v)} with {len(v)} elements")
                else:
                    print(f"      {k}: {type(v)}")
    
    print("\n" + "=" * 80)
    print(f"Successfully loaded {len(loaded_data)} checkpoint files")
    
    # Summary of loaded data
    print("\nSummary of loaded data:")
    for key, info in loaded_data.items():
        params = info['params']
        print(f"  {key}: N_hmc={params['N_hmc']}, Ltau={params['Ltau']}, K={params['K']}, Nrv={params['Nrv']}")

    
    import matplotlib.pyplot as plt

    # Extract the data array; confirm key exists
    key = 'N30_Ltau300'
    if key in loaded_data:
        spsm_r_tau_arr = loaded_data[key]['data']['spsm_r_tau_list']  # Expect shape (Ncfg, Nbs, Ntau, ?, ?)
        y = spsm_r_tau_arr[0, 0, :, 0, 2]  # shape: (Ntau,)
        x = range(len(y))
        plt.figure()
        plt.plot(x, y, marker="o", linestyle='-', label='spsm_r_tau_list[0,0,:,0,0]')
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("tau index")
        plt.ylabel("Value")
        plt.title("Plot of spsm_r_tau_list[0,0,:,0,0] for N=30, Ltau=300")
        plt.legend()
        plt.show()
    else:
        print(f"Key {key} not in loaded_data")
    
    return loaded_data


if __name__ == "__main__":
    # Load all checkpoint data
    data = main()
    
    # Make data available in global scope for interactive use
    globals().update(data)