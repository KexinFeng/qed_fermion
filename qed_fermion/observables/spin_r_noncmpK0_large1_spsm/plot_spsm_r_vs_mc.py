import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set up script path and HMC data folder (same as plot_fit.py)
script_path = os.path.dirname(os.path.abspath(__file__))
hmc_folder = "/Users/kx/Desktop/hmc/fignote/cmp_noncmp_result/noncmpK0_large1_spsm/hmc_check_point_noncmpK0_large1_spsm"

# Choose lattice size (example: Lx=20)
Lx = 40
Ltau = int(10 * Lx)
hmc_file = f"ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_10000_bs2_Jtau_1.2_K_0_dtau_0.1_delta_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_cmp_False_step_10000.pt"
hmc_filename = os.path.join(hmc_folder, hmc_file)

if not os.path.exists(hmc_filename):
    raise FileNotFoundError(f"File not found: {hmc_filename}")

# Load checkpoint data
res = torch.load(hmc_filename, map_location='cpu')
spsm_r = res['spsm_r_list']  # Shape: [timesteps, batch_size, Ly, Lx]

# MC step indices (skip initial equilibration)
start = 0
hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
end = int(hmc_match.group(1))
seq_idx = np.arange(start, end, 1)

# Plot spsm_r at different distances as a function of MC steps
plt.figure(figsize=(8, 5))
for x in [3, 5]:
    # Average over batch, y=0
    y = 0
    spsm_r_mean = spsm_r[seq_idx, :, y, x].abs().mean(axis=1).numpy()
    plt.plot(seq_idx, spsm_r_mean, label=f'spsm[{x}]')

# plt.yscale('log')

plt.xlabel("MC Step")
plt.ylabel("|Spsm_r|")
plt.title("spsm_r Over MC Steps (Lx=20)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
save_dir = os.path.join(script_path, "./figures/spsm_r_vs_mc")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "spsm_r_vs_mc_steps.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

plt.show()
