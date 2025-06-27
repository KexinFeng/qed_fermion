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
sys.path.insert(0, script_path + '/../../../')
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


def postprocess_and_write_spsm(bosons, output_dir, Lx, Ly, Ltau, bid=1, Nrv=10, mxitr=200, start=5000):
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

    os.makedirs(output_dir, exist_ok=True)
    boson_seq = bosons.view(bosons.shape[0], bosons.shape[1], 2, Lx, Ly, Ltau)[start:].to(se.device)
    spsm_k = []
    DD_k = []
    for boson in tqdm(boson_seq):  # boson: [J/bs, 2, Lx, Ly, Ltau]
        eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]
        if se.cuda_graph_se:
            obsr_se = se.graph_runner(boson.to(se.device), eta)
        else:
            obsr_se = se.get_fermion_obsr(boson.to(se.device), eta)

        obsr_gt = se.get_fermion_obsr_gt(boson.to(se.device))
        # spsm_k.append(obsr['spsm_k_abs'].cpu().numpy())
        # spsm_k_gt = obsr_gt['spsm_k_abs'].cpu().numpy()
        # spsm_k.append(spsm_k_gt)
        # spsm_k.append(spsm_k_gt)
        DD_k.append(obsr_gt['DD_k_abs'].cpu().numpy())

        # spsm_r = obsr['spsm_r'].cpu().numpy()  # [seq, J/bs, Ly, Lx]
        # spsm_r_gt = obsr_gt['spsm_r'].cpu().numpy()  # [seq, J/bs, Ly, Lx]
        dbstop = 1
        
    # spsm_k = np.array(spsm_k)  # [seq, J/bs, Ly, Lx]
    DD_k = np.array(DD_k)  # [seq, J/bs, Ly, Lx]
    # spsm_k_mean = spsm_k.mean(axis=0)  # [J/bs, Ly, Lx]
    # spsm_k_std = spsm_k.std(axis=0)  # [J/bs, Ly, Lx]
    DD_k_mean = DD_k.mean(axis=0)  # [J/bs, Ly, Lx]
    DD_k_std = DD_k.std(axis=0)  # [J/bs, Ly, Lx]

    output_file = os.path.join(output_dir, f"spsm_k_b{bid}.pt")
    torch.save({'DD_k_mean': DD_k_mean, 
                'DD_k_std': DD_k_std}, 
                output_file)
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    # Create configs file
    Lx = int(os.getenv("Lx", '6'))
    print(f"Lx: {Lx}")
    Ltau = int(os.getenv("Ltau", '10'))
    print(f"Ltau: {Ltau}")
    Nrv = int(os.getenv("Nrv", '200'))
    print(f"Nrv: {Nrv}")
    mxitr = int(os.getenv("mxitr", '400'))
    print(f"mxitr: {mxitr}")


    Js = [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
    Js = [3.0]
    # Js = [0.5, 1.0, 3.0]

    bs = 5

    # input_folder = "/Users/kx/Desktop/hmc/fignote/equilibrium_issue/hmc_check_point_bench/"
    # input_folder = "/home/fengx463/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_bench/"
    # input_folder = "./qed_fermion/check_points/hmc_check_point_bench/"
    # input_folder = "/home/fengx463/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point_bench_6810_2/"
    # input_folder = "/users/4/fengx463/hmc/fignote/equilibrum_issue/"
    input_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/benchmark_6x6x10_bs5/hmc_check_point_6x10/"
    # input_folder = "/users/4/fengx463/hmc/fignote/hmc_check_point_6x10/"

    start = -1
    end = 10000

    @time_execution
    def iterate_func():
        for bid in range(bs):
            if bid > 0: continue
            bosons = []
            for J in Js:

                hmc_filename = f"/stream_ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_6000_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.05_N_leapfrog_4_m_1_step_6000.pt"

                # Load and write
                # [seq, bs, 2*Lx*Ly*Ltau]
                boson_seq = torch.load(input_folder + hmc_filename)
                # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
                print(f'Loaded: {hmc_filename}')  
                
                # Post-process and write spsm
                Ly = Lx  # Assuming square lattice, adjust if not
                
                bosons.append(boson_seq[:, bid])

            bosons_tnsr = torch.stack(bosons, dim=1).to(device)  # [seq, J/bs, 2*Lx*Ly*Ltau]
            
            output_dir = script_path + f"/data_inv_start{start}/Lx_{Lx}_Ltau_{Ltau}_Nrv_{Nrv}_mxitr_{mxitr}/"
            postprocess_and_write_spsm(bosons_tnsr, output_dir, Lx, Ly, Ltau, Nrv=Nrv, mxitr=mxitr, start=start, bid=bid)
            
    iterate_func()


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
