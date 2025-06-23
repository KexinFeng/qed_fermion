import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ion()
import numpy as np
# matplotlib.use('MacOSX')
from matplotlib import rcParams
rcParams['figure.raise_window'] = False
import os
script_path = os.path.dirname(os.path.abspath(__file__))
import torch
import sys
sys.path.insert(0, script_path + '/../../../../')
import time
from qed_fermion.hmc_sampler_batch import HmcSampler
from qed_fermion.stochastic_estimator import StochaticEstimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds\n")
        return result
    return wrapper


def postprocess_and_write_spsm(bosons, output_dir, Lx, Ly, Ltau, Nrv=10, mxitr=200, start=5000):
    """
    boson: [seq, J/bs, 2 * Lx * Ly * Ltau]
    """
    hmc = HmcSampler()
    hmc.Lx = Lx
    hmc.Ly = Ly
    hmc.Ltau = Ltau
    hmc.bs = bosons.shape[1]
    hmc.reset()

    se = StochaticEstimator(hmc, cuda_graph_se=hmc.cuda_graph)
    se.Nrv = Nrv
    se.max_iter_se = mxitr
    if se.cuda_graph_se:
        se.init_cuda_graph()
    # eta = se.random_vec_bin()
    os.makedirs(output_dir, exist_ok=True)
    boson_conf = bosons.view(bosons.shape[0], bosons.shape[1], 2, Lx, Ly, Ltau)[start:]
    spsm_k = []
    DD_k = []
    for boson in tqdm(boson_conf):  # boson: [J/bs, 2, Lx, Ly, Ltau]
        eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]
        if se.cuda_graph_se:
            obsr = se.graph_runner(boson.to(se.device), eta)
        else:
            obsr = se.get_fermion_obsr(boson.to(se.device), eta)
        spsm_k.append(obsr['spsm_k_abs'].cpu().numpy())
        DD_k.append(obsr['DD_k'].cpu().numpy())
        
    spsm_k = np.array(spsm_k)  # [seq, J/bs, Ly, Lx]
    DD_k = np.array(DD_k)  # [seq, J/bs, Ly, Lx]
    spsm_k_mean = spsm_k.mean(axis=0)  # [J/bs, Ly, Lx]
    spsm_k_std = spsm_k.std(axis=0)  # [J/bs, Ly, Lx]
    DD_k_mean = DD_k.mean(axis=0)  # [J/bs, Ly, Lx]
    DD_k_std = DD_k.std(axis=0)  # [J/bs, Ly, Lx]

    output_file = os.path.join(output_dir, "spsm_k.pt")
    torch.save({'mean': spsm_k_mean, 
                'std': spsm_k_std, 
                'DD_k_mean': DD_k_mean, 
                'DD_k_std': DD_k_std}, 
                output_file)
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    # Create configs file
    Lx = int(os.getenv("Lx", '10'))
    print(f"Lx: {Lx}")
    Ltau = int(os.getenv("Ltau", '100'))
    print(f"Ltau: {Ltau}")
    Nrv = int(os.getenv("Nrv", '100'))
    print(f"Nrv: {Nrv}")
    mxitr = int(os.getenv("mxitr", '400'))
    print(f"mxitr: {mxitr}")


    Js = [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
    # Js = [0.5, 1.0, 3.0]

    bs = 2

    input_folder = "/Users/kx/Desktop/hmc/fignote/equilibrium_issue/hmc_check_point_bench/"
    input_folder = "/home/fengx463/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_bench/"
    input_folder = "./qed_fermion/check_points/hmc_check_point_bench/"
    input_folder = "/home/fengx463/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_bench_6810_2/"
    input_folder = "/users/4/fengx463/hmc/fignote/equilibrum_issue/"

    start = -50
    end = 10000

    @time_execution
    def iterate_func():
        bosons = []
        for J in Js:

            bid = 1

            hmc_filename = f"/stream_ckpt_N_t_hmc_{Lx}_Ltau_{Ltau}_Nstp_{end}_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.028_N_leapfrog_5_m_1_cg_rtol_1e-09_max_block_idx_1_gear0_steps_1000_dt_deque_max_len_5_step_{end}.pt"

            # Load and write
            # [seq, bs, 2*Lx*Ly*Ltau]
            boson_seq = torch.load(input_folder + hmc_filename)
            # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
            print(f'Loaded: {hmc_filename}')  
            
            # Post-process and write spsm
            Ly = Lx  # Assuming square lattice, adjust if not
            
            bosons.append(boson_seq[:, bid])

        bosons = torch.stack(bosons, dim=1).to(device)  # [seq, J/bs, 2*Lx*Ly*Ltau]
        
        output_dir = script_path + f"/data_se_start{start}/Lx_{Lx}_Ltau_{Ltau}_Nrv_{Nrv}_mxitr_{mxitr}/"
        postprocess_and_write_spsm(bosons, output_dir, Lx, Ly, Ltau, Nrv=Nrv, mxitr=mxitr, start=start)
            
  
    iterate_func()
