import collections
import gc
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm
import sys
import os
import matlab.engine
import concurrent.futures

# matplotlib.use('MacOSX')
plt.ion()
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from matplotlib import rcParams
rcParams['figure.raise_window'] = False
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')
if torch.cuda.is_available():
    from qed_fermion import _C 

from qed_fermion.utils.coupling_mat3 import initialize_curl_mat
from qed_fermion.post_processors.load_write2file_convert import time_execution
from qed_fermion.force_graph_runner import ForceGraphRunner

BLOCK_SIZE = (4, 8)
print(f"BLOCK_SIZE: {BLOCK_SIZE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f"device: {device}")

dtype = torch.float32
cdtype = torch.complex64
cg_dtype = torch.complex64
print(f"dtype: {dtype}")
print(f"cdtype: {cdtype}")
print(f"cg_cdtype: {cg_dtype}")

debug_mode = int(os.getenv("debug", '0')) != 0
print(f"debug_mode: {debug_mode}")
if not debug_mode:
    import matplotlib
    matplotlib.use('Agg') # write plots to disk without requiring a display or GUI.

mass_mode = int(os.getenv("mass_mode", '0')) # 1: mass ~ inverse sigma; -1: mass ~ sigma
print(f"mass_mode: {mass_mode}")
cuda_graph = int(os.getenv("cuda_graph", '0')) != 0
print(f"cuda_graph: {cuda_graph}")

dt_deque_max_len = 10
sigma_mini_batch_size = 10 if debug_mode else 100
print(f"dt_deque_max_len: {dt_deque_max_len}")
print(f"sigma_mini_batch_size: {sigma_mini_batch_size}")

start_total_monitor = 10 if debug_mode else 100
start_load = 2000

# Set a random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

executor = None

def initialize_executor():
    global executor
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return executor

class HmcSampler(object):
    def __init__(self, Lx=6, Ltau=10, J=0.5, Nstep=3000, config=None):
        # Dims
        self.Lx = Lx
        self.Ly = Lx
        self.Ltau = Ltau
        self.bs = 2 if torch.cuda.is_available() else 1
        print(f"bs: {self.bs}")
        self.Vs = self.Lx * self.Ly

        # Couplings
        self.Nf = 2
        # J = 0.5  # 0.5, 1, 3
        # self.dtau = 2/(J*self.Nf)

        self.dtau = 0.1
        scale = self.dtau  
        # Lagrangian * dtau
        self.J = J / scale * self.Nf / 4
        self.K = 1 * scale * self.Nf * 1/2
        self.t = 1

        # t * dtau < const
        # fix t, increase J

        # J,t = (4, 0.1) 40:  Ff=4, Fb=9.7, delta_t=0.2
        # J,t = (20, 0.5) 40: Ff=4, Fb=9.7, delta_t=0.2
        # J,t = (40, 1) 40:   Ff=4, Fb=1, delta_t=2

        self.boson = None

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y
        self.plt_rate = 5 if debug_mode else max(start_total_monitor, 500)
        self.ckp_rate = 500
        self.stream_write_rate = Nstep
        self.memory_check_rate = 10 if debug_mode else 1000

        # Statistics
        self.N_step = Nstep
        self.step = 0
        self.cur_step = 0

        self.G_list = torch.zeros(self.N_step, self.bs, self.num_tau + 1)
        self.accp_list = torch.zeros(self.N_step, self.bs, dtype=torch.bool)
        self.accp_rate = torch.zeros(self.N_step, self.bs)
        self.S_plaq_list = torch.zeros(self.N_step, self.bs)
        self.S_tau_list = torch.zeros(self.N_step, self.bs)
        # self.Sf_list = torch.zeros(self.N_step, self.bs)
        # self.H_list = torch.zeros(self.N_step, self.bs)

        self.cg_iter_list = torch.zeros(self.N_step, self.bs)
        self.cg_r_err_list = torch.zeros(self.N_step, self.bs)
        self.delta_t_list = torch.zeros(self.N_step, self.bs)

        # boson_seq
        self.boson_seq_buffer = torch.zeros(self.stream_write_rate, self.bs, 2*self.Lx*self.Ly*self.Ltau, device='cpu', dtype=dtype)

        # Leapfrog
        self.debug_pde = False
        # self.m = 1/2 * 4 / scale * 0.05
        self.m = 1

        # self.delta_t = 0.05 # (L=6)
        # self.delta_t = 0.2/4
        # self.delta_t = 0.005
        self.delta_t = 0.04/2
        # self.delta_t = 0.0066
        # self.delta_t = 0.04/4
        self.delta_t = 0.05/4 # (L=8)
        self.delta_t = 0.14/8
        # self.delta_t = 0.0175 = 0.07/4
        self.delta_t = 0.025  # >=0.03 will trap the leapfrog at the beginning
        # self.delta_t = 0.008 # (L=10)
        self.delta_t = 0.06/4 if self.Lx == 8 else 0.08

        self.delta_t = 0.028
        # self.m = (self.delta_t / 0.025) ** 2
        # self.delta_t = 0.2 * (self.m / 50)**(1/2)
        self.delta_t = 0.1 * (self.m / 50)**(1/2)

        # self.delta_t = 0.03 # TODO: Does increasing inverse-matvec_mul accuracy help with the acceptance rate / threshold? If so, the bottelneck is at the accuracyxs
        # self.delta_t = 0.1 # This will be too large and trigger H0,Hfin not equal, even though N_leapfrog is cut half to 3
        # For the same total_t, the larger N_leapfrog, the smaller error and higher acceptance.
        # So for a given total_t, there is an optimal N_leapfrog which is the smallest N_leapfrog s.t. the acc is larger than say 0.9 the saturate accp (which is 1).
        # Then increasing total_t will increase N_leapfrog*, if total_t reasonable.
        # So proper total_t is a function of N_leapfrog* for a given threshold like 0.9.
        # So natually an adaptive optimization algorithm can be obtained: for a fixed N_leapfrog, check the acceptance_rate / acc_threshold and adjust total_t.

        self.delta_t_tensor = torch.full((self.bs,), self.delta_t, device=device, dtype=dtype)

        # self.N_leapfrog = 6 # already oscilates back
        # self.N_leapfrog = 8
        # self.N_leapfrog = 2
        # self.N_leapfrog = 6
        self.N_leapfrog = 5

        self.threshold_queue = [collections.deque(maxlen=dt_deque_max_len) for _ in range(self.bs)]

        self.lower_limit = 0.5
        self.upper_limit = 0.8

        # Sigma adaptive mass
        self.sigma_hat = torch.ones(self.bs, 2, self.Lx, self.Ly, self.Ltau // 2 + 1, device=device, dtype=dtype)
        self.sigma_hat_cpu = torch.ones(self.bs, 2, self.Lx, self.Ly, self.Ltau // 2 + 1, device='cpu', dtype=dtype)
        self.boson_mean_cpu = torch.zeros(self.bs, 2, self.Lx, self.Ly, self.Ltau, device='cpu', dtype=dtype)
        self.sigma_mini_batch_size = sigma_mini_batch_size
        self.sigma_hat_mini_batch_last_i = 0
        self.annealing_step = 1000 if not debug_mode else 50

        self.multiplier = torch.full_like(self.sigma_hat, 2)
        self.multiplier[..., torch.tensor([0, -1], dtype=torch.int64, device=self.sigma_hat.device)] = 1
        
        self.lmd = 0.95
        self.sig_min = 0.8
        self.sig_max = 1.2

        # CG
        self.cg_rtol = 1e-7
        self.cg_rtol = 1e-5
        self.cg_rtol_tensor = torch.tensor([self.cg_rtol], dtype=dtype, device=device)
        # self.max_iter = 400  # at around 450 rtol is so small that becomes nan
        # self.cg_rtol = 1e-9
        self.max_iter = 1000
        print(f"cg_rtol: {self.cg_rtol} max_iter: {self.max_iter}")
        self.precon = None
        self.plt_cg = False
        self.verbose_cg = False
        self.use_cuda_kernel = torch.cuda.is_available()
        if self.use_cuda_kernel:
            pass
            # assert self.bs < 2
        
        # CUDA Graph for force_f_fast
        self.cuda_graph = cuda_graph  # Disable CUDA graph support for now
        print(f'cuda_graph:{ self.cuda_graph}')
        self.force_graph_runners = {}
        self.force_graph_memory_pool = None
        # self._MAX_ITERS_TO_CAPTURE = [300, 500, 800, 1000]
        self._MAX_ITERS_TO_CAPTURE = [200]
        if self.cuda_graph:
            self.max_iter = self._MAX_ITERS_TO_CAPTURE[0]

        # Debug
        torch.manual_seed(0)

        # Initialization
        self.reset()
    
    def reset(self):
        self.Vs = self.Lx * self.Ly
        self.initialize_curl_mat()
        self.initialize_geometry()
        self.initialize_specifics()
        self.initialize_boson_time_slice_random_uniform()

    def initialize_specifics(self):      
        self.specifics = f"hmc_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_bs{self.bs}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf*2:.2g}_dtau_{self.dtau:.2g}_delta_t_{self.delta_t:.2g}_N_leapfrog_{self.N_leapfrog}_m_{self.m:.2g}_cg_rtol_{self.cg_rtol:.2g}_lmd_{self.lmd:.2g}_sig_min_{self.sig_min:.2g}_sig_max_{self.sig_max:.2g}_lower_limit_{self.lower_limit:.2g}_upper_limit_{self.upper_limit:.2g}_mass_mode_{mass_mode}"

    def get_specifics(self):
        return f"hmc_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_bs{self.bs}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf*2:.2g}_dtau_{self.dtau:.2g}_delta_t_{self.delta_t:.2g}_N_leapfrog_{self.N_leapfrog}_m_{self.m:.2g}_cg_rtol_{self.cg_rtol:.2g}_lmd_{self.lmd:.2g}_sig_min_{self.sig_min:.2g}_sig_max_{self.sig_max:.2g}_lower_limit_{self.lower_limit:.2g}_upper_limit_{self.upper_limit:.2g}_mass_mode_{mass_mode}"

    def initialize_force_graph(self):
        """Initialize CUDA graph for force_f_fast function."""
        if not self.cuda_graph:
            return
            
        print("Initializing CUDA graph for force_f_fast...")
        
        dummy_psi = torch.zeros(self.bs, self.Lx * self.Ly * self.Ltau, 
                               dtype=cdtype, device=device)
        dummy_boson = torch.zeros(self.bs, 2, self.Lx, self.Ly, self.Ltau, 
                                 dtype=dtype, device=device)
        dummy_rtol = self.cg_rtol_tensor
        
        # Capture graphs for different batch sizes
        for max_iter in reversed(self._MAX_ITERS_TO_CAPTURE):
            # Capture graphs for given batch size
            graph_runner = ForceGraphRunner(self)
            graph_memory_pool = graph_runner.capture(
                dummy_psi,
                dummy_boson,
                dummy_rtol,
                max_iter,
                self.force_graph_memory_pool
            )
            
            # Store the graph runner and memory pool
            self.force_graph_runners[max_iter] = graph_runner
            self.force_graph_memory_pool = graph_memory_pool
                
        print(f"CUDA graph initialization complete for batch sizes: {self._MAX_ITERS_TO_CAPTURE}")

    
    def reset_precon(self):
        # Check if preconditioner file exists
        data_folder = script_path + "/preconditioners/"
        file_name = f"precon_ckpt_L_{self.Lx}_Ltau_{self.Ltau}_dtau_{self.dtau}_t_{self.t}"
        file_path = os.path.join(data_folder, file_name + ".pt")

        precon_dict = None
        if not os.path.exists(file_path): 
            @time_execution     
            def embedded_func():    
                print(f"Preconditioner file {file_path} does not exist. \nComputing the preconditioner.....")
                # Compute preconditioner if not exists
                curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
                boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
                boson = boson.repeat(1 * self.Ltau, 1)
                pi_flux_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
                precon_dict = self.get_precon2(pi_flux_boson)
                gc.collect()

                # Filter precon
                print("# Filter precon")
                band_width1 = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64) * self.Vs
                band_width2 = (torch.tensor([-1, -2, -3, -4, -5], device=device, dtype=torch.int64) + self.Ltau) * self.Vs
                band_width = torch.cat([band_width1, band_width2])
                dist = (precon_dict['indices'][0] - precon_dict['indices'][1]).abs().to(device)
                # Filter entries whose dist is in band_width
                mask = torch.isin(dist, band_width)

                print(f"mask_sum= {mask.sum().item()}, active precon entries: {len(precon_dict['values'])}")

                filtered_indices = precon_dict['indices'].to(device)[:, mask]
                filtered_values = precon_dict['values'].to(device)[mask]

                # Create a new sparse tensor with the filtered entries
                filtered_precon = torch.sparse_coo_tensor(
                    filtered_indices,
                    filtered_values,
                    size=precon_dict["size"],
                    dtype=cdtype,
                    device=device
                ).coalesce()

                # Save preconditioner to file
                precon_dict = {
                    "indices": filtered_precon.indices().cpu(),
                    "values": filtered_precon.values().cpu(),
                    "size": filtered_precon.size()
                }
                self.save_to_file(precon_dict, data_folder, file_name)

                self.precon = filtered_precon.to_sparse_csr()
                return precon_dict
            
            precon_dict = embedded_func()
            exit(0)
            
        else:
            # Load preconditioner from file
            precon_dict = torch.load(file_path)
            print(f"Loaded preconditioner from {file_path}")

            indices = precon_dict["indices"].to(device)
            values = precon_dict["values"].to(device)
            
            # Create a new sparse tensor with the filtered entries
            precon = torch.sparse_coo_tensor(
                indices,
                values,
                size=precon_dict["size"],
                dtype=cdtype,
                device=device
            ).coalesce()

            self.precon = precon
            self.precon_csr = self.precon.to_sparse_csr()

 
    @staticmethod
    def filter_mat(mat, M):
        if not mat.is_coalesced():
            mat = mat.coalesce()
        abs_values = torch.abs(mat.values())
        filter_mask = abs_values >= 1e-4
        mat_indices = mat.indices()[:, filter_mask]
        mat_values = mat.values()[filter_mask]
        del mat
        # gc.collect()
        mat = torch.sparse_coo_tensor(
            mat_indices,
            mat_values,
            (M.size(0), M.size(1)),
            dtype=M.dtype,
            device=M.device
        ).coalesce() 
        return mat    
    
    def get_precon2(self, pi_flux_boson, output_scipy=False):
        iter = 20
        thrhld = 0.1 
        diagcomp = 0.05  

        MhM, _, _, M = self.get_M_sparse(pi_flux_boson)
        retrieved_indices = M.indices() + 1  # Convert to 1-based indexing for MATLAB
        retrieved_values = M.values()

        # Pass indices and values to MATLAB
        matlab_function_path = script_path + '/utils/'
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_function_path)

        # Convert indices and values directly to MATLAB format
        matlab_indices = matlab.double(retrieved_indices.cpu().tolist())
        matlab_values = matlab.double(retrieved_values.cpu().tolist(), is_complex=True)

        # Call MATLAB function
        result_indices_i, result_indices_j, result_values = eng.ichol_m(
            matlab_indices, matlab_values, M.size(0), M.size(1), nargout=3
        )
        eng.quit()

        # Convert MATLAB results directly to PyTorch tensors
        result_indices_i = torch.tensor(result_indices_i, dtype=torch.long, device=M.device).view(-1) - 1
        result_indices_j = torch.tensor(result_indices_j, dtype=torch.long, device=M.device).view(-1) - 1
        result_values = torch.tensor(result_values, dtype=M.dtype, device=M.device).view(-1)

        # Create MhM_inv as a sparse_coo_tensor
        M_pc = torch.sparse_coo_tensor(
            torch.stack([result_indices_i, result_indices_j]),
            result_values,
            (M.size(0), M.size(1)),
            dtype=M.dtype,
            device=M.device
        ).coalesce()

        # Diagonal matrix from M_pc
        diag_values = M_pc.values()[M_pc.indices()[0] == M_pc.indices()[1]]
        dd_inv = 1.0 / diag_values
        del diag_values
        I = torch.sparse_coo_tensor(
            torch.arange(M.size(0), device=M.device).repeat(2, 1),
            torch.ones(M.size(0), device=M.device, dtype=M.dtype),
            (M.size(0), M.size(0)),
            device=M.device,
            dtype=M.dtype
        ).coalesce()

        # Neumann series approximation for matrix inverse
        dd_inv = torch.sparse_coo_tensor(
            torch.arange(M.size(0), device=M.device).repeat(2, 1),
            dd_inv,
            (M.size(0), M.size(0)),
            dtype=M.dtype,
            device=M.device
        ).coalesce()
        gc.collect() 

        M_diag_scaled = torch.sparse.mm(M_pc, dd_inv)
        del M_pc
        gc.collect() 

        M_itr = I - M_diag_scaled
        M_temp = I.clone()
        M_inv = I
        for i in tqdm(range(iter), desc="Neumann Series Iteration"):
            M_temp = torch.sparse.mm(M_temp, M_itr)
            M_inv = M_inv + M_temp
            if i % math.floor(iter * 0.34) == 0:
                M_temp = self.filter_mat(M_temp, M)
                M_inv = self.filter_mat(M_inv, M)
                gc.collect()
                
        gc.collect() 
        del M_temp
        del M_itr

        # Filter small elements right after Neumann series
        print("# Filter small elements right after Neumann series")
        M_inv = self.filter_mat(M_inv, M) 
        gc.collect()     

        # Scale by the inverse diagonal
        print('# Scale by the inverse diagonal')
        M_inv = torch.sparse.mm(dd_inv, M_inv)
        del dd_inv
        gc.collect()

        # Compute O_inv1 = M_inv' * M_inv
        print("# Compute O_inv1 = M_inv' * M_inv")
        O_inv1 = torch.sparse.mm(M_inv.T.conj(), M_inv)
        del M_inv

        # Filter small elements to maintain sparsity
        print("# Filter small elements to maintain sparsity")
        abs_values = torch.abs(O_inv1.values())
        filter_mask = abs_values >= thrhld

        # Gather filtered indices and values
        O_inv_indices = O_inv1.indices()[:, filter_mask]
        O_inv_values = O_inv1.values()[filter_mask]

        precon_dict = {
                    "indices": O_inv_indices,
                    "values": O_inv_values,
                    "size": O_inv1.shape
                }
        return precon_dict

    def adjust_delta_t(self):
        """
        Adjust delta_t for each batch element based on its acceptance rate.
        """
        lower_limit = self.lower_limit
        upper_limit = self.upper_limit

        for b in range(self.bs):
            if len(self.threshold_queue[b]) < self.threshold_queue[b].maxlen:
                continue
            avg_accp_rate = sum(self.threshold_queue[b]) / len(self.threshold_queue[b])

            if debug_mode:
                print(f'Batch {b} avg_clipped_threshold: {avg_accp_rate:.4f}')
            old_delta_t = self.delta_t_tensor[b].item()

            if avg_accp_rate < lower_limit:
                self.delta_t_tensor[b] *= 0.9
            elif upper_limit < avg_accp_rate:
                self.delta_t_tensor[b] *= 1.1
            else:
                continue

            if debug_mode:
                print(f"----->Adjusted delta_t[{b}] from {old_delta_t:.4f} to {self.delta_t_tensor[b].item():.4f}")
            self.threshold_queue[b].clear()

    @staticmethod
    def apply_preconditioner(r, MhM_inv=None, matL=None):
        if MhM_inv is None and matL is None:
            return r
        if MhM_inv is not None:
            MhM_inv = MhM_inv.to(r.dtype)
            return torch.sparse.mm(MhM_inv, r)
        
        matL = matL.to(r.dtype)
        # Apply incomplete Cholesky preconditioner via SpSV solver
        # Convert sparse tensor to scipy sparse matrix
        matL_scipy = sp.csr_matrix((matL.values().cpu().numpy(),
                    (matL.indices()[0].cpu().numpy(),
                        matL.indices()[1].cpu().numpy())),
                    shape=matL.shape, dtype=matL.values().cpu().numpy().dtype)
        r_numpy = r.cpu().numpy()

        # Solve using scipy's sparse triangular solver
        tmp = splinalg.spsolve_triangular(matL_scipy, r_numpy, lower=True)
        output = splinalg.spsolve_triangular(matL_scipy.T.conj(), tmp, lower=False)
        
        # Convert back to torch tensor
        return torch.tensor(output, dtype=r.dtype, device=r.device)  


    def preconditioned_cg_fast_test(self, boson, b, MhM_inv=None, matL=None, rtol_tensor=None, max_iter=400, b_idx=None, axs=None, cg_dtype=torch.complex64, MhM=None):
        """
        Solve M'M x = b using preconditioned conjugate gradient (CG) algorithm.

        :param boson: boson.permute([0, 4, 3, 2, 1]).view(self.bs, self.Ltau, -1)
        :param b: Right-hand side vector, [bs, Ltau*Ly*Lx]
        :param tol: Tolerance for convergence
        :param max_iter: Maximum number of iterations
        :return: Solution vector x, [bs, Ltau*Ly*Lx]
        """
        assert b.dtype == torch.complex64, "Expected b to have dtype torch.complex64"
        assert len(b.shape) == 2
        assert len(boson.shape) == 3
        boson = boson.view(self.bs, -1)
        b = b.view(self.bs, -1)
        norm_b = torch.norm(b, dim=1)

        active_bs = torch.full((self.bs, 1), 1, device=device, dtype=b.dtype)

        # Initialize variables
        x = torch.zeros_like(b).view(self.bs, -1)
        r = b.view(self.bs, -1) - _C.mhm_vec(boson, x, self.Lx, self.dtau, *BLOCK_SIZE)
        z = _C.precon_vec(r, self.precon_csr, self.Lx)

        p = z
        rz_old = torch.einsum('bj,bj->b', r.conj(), z).real

        residuals = []

        if self.plt_cg and axs is not None:
            # Plot intermediate results
            line_res = []
            for b_idx in range(self.bs):
                line = axs.plot(residuals, marker='o', linestyle='-', label=b_idx, color=f'C{b_idx}')
                line_res.append(line[0])
            axs.set_ylabel('Residual Norm')
            axs.set_yscale('log')
            axs.legend()
            axs.grid(True)
            axs.set_title(f'{self.Lx}x{self.Ly}x{self.Ltau} Lattice b_idx={b_idx}')

            plt.pause(0.01)  # Pause to update the plot

        cnt = 0
        iterations = torch.zeros(self.bs, dtype=torch.int64, device=device) if not self.cuda_graph else None

        for i in range(max_iter):
            # Matrix-vector product with M'M
            Op = _C.mhm_vec(boson, p, self.Lx, self.dtau, *BLOCK_SIZE)

            alpha = (rz_old / torch.einsum('bj,bj->b', p.conj(), Op).real).unsqueeze(-1)
            x += alpha * p * active_bs
            r -= alpha * Op * active_bs

            # Compute and store the error (norm of the residual)
            error = torch.norm(r, dim=1) / norm_b
            residuals.append(error)

            if self.plt_cg and axs is not None:
                # Plot intermediate results
                for b_idx in range(self.bs):
                    ys = [residuals[j][b_idx].item() for j in range(len(residuals))]
                    line_res[b_idx].set_data(range(len(residuals)), ys)
                axs.relim()
                axs.autoscale_view()
                plt.pause(0.01)  # Pause to update the plot

                method_name = "pcg_fast"
                save_dir = os.path.join(script_path, f"./figures_pcg/")
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"{method_name}.pdf")
                plt.savefig(file_path, format="pdf", bbox_inches="tight")
                print(f"Figure saved at: {file_path}")

            # Check for convergence
            active_bs = torch.where(error.view(-1, 1) > rtol_tensor, active_bs, torch.zeros_like(active_bs))

            if not self.cuda_graph:
                if (active_bs == 0).any():
                    if self.verbose_cg:
                        print(f"{torch.nonzero(active_bs).tolist()} Converged in {i+1} iterations.")
                    if (active_bs == 0).all():
                        break
                
                iterations += (active_bs == 1).view(-1).long()

            # z = torch.sparse.mm(MhM_inv, r) if MhM_inv is not None else r  # Apply preconditioner to rtL)
            z = _C.precon_vec(r, self.precon_csr, self.Lx)
            rz_new = torch.einsum('bj,bj->b', r.conj(), z).real
            beta = rz_new / rz_old
            p = z + beta.unsqueeze(-1) * p * active_bs
            rz_old = rz_new

            cnt += 1

        return x, iterations, error

    def preconditioned_cg(self, MhM, b, MhM_inv=None, matL=None, rtol=1e-8, max_iter=100, b_idx=None, axs=None, cg_dtype=torch.complex64):
        """
        Solve M'M x = b using preconditioned conjugate gradient (CG) algorithm.

        :param M: Sparse matrix M from get_M_sparse()
        :param b: Right-hand side vector
        :param tol: Tolerance for convergence
        :param max_iter: Maximum number of iterations
        :return: Solution vector x
        """
        dtype_init = MhM.dtype
        MhM = MhM.to(cg_dtype)
        b = b.to(cg_dtype)
        if MhM_inv is not None:
            MhM_inv = MhM_inv.to(cg_dtype)

        # Initialize variables
        x = torch.zeros_like(b).view(-1, 1)
        r = b.view(-1, 1) - torch.sparse.mm(MhM, x)
        # z = torch.sparse.mm(MhM_inv, r) if MhM_inv is not None else r
        z = self.apply_preconditioner(r, MhM_inv, matL)
        p = z
        rz_old = torch.dot(r.conj().view(-1), z.view(-1)).real

        errors = []
        residuals = []

        if self.plt_cg and axs is not None:
            # Plot intermediate results
            line_res = axs.plot(residuals, marker='o', linestyle='-', label='˚no precon.' if MhM_inv is None and matL is None else 'precon.', color=f'C{0}' if MhM_inv is None and matL is None else f'C{1}')
            axs.set_ylabel('Residual Norm')
            axs.set_yscale('log')
            axs.legend()
            axs.grid(True)
            axs.set_title(f'{self.Lx}x{self.Ly}x{self.Ltau} Lattice b_idx={b_idx}')

            plt.pause(0.01)  # Pause to update the plot

        cnt = 0
        for i in range(max_iter):
            # Matrix-vector product with M'M
            Op = torch.sparse.mm(MhM, p)
            
            alpha = rz_old / torch.dot(p.conj().view(-1), Op.view(-1)).real
            x += alpha * p
            r -= alpha * Op

            # Compute and store the error (norm of the residual)
            error = torch.norm(r).item() / torch.norm(b).item()
            errors.append(error)
            residuals.append(error)

            if self.plt_cg and axs is not None:
                # Plot intermediate results
                line_res[0].set_data(range(len(residuals)), residuals)
                axs.relim()
                axs.autoscale_view()
                plt.pause(0.01)  # Pause to update the plot

                method_name = "pcg"
                save_dir = os.path.join(script_path, f"./figures_pcg/")
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"{method_name}.pdf")
                plt.savefig(file_path, format="pdf", bbox_inches="tight")
                print(f"Figure saved at: {file_path}")

            # Check for convergence
            if error < rtol:
                if self.verbose_cg:
                    print(f"Converged in {i+1} iterations.")
                break

            # z = torch.sparse.mm(MhM_inv, r) if MhM_inv is not None else r  # Apply preconditioner to r
            z = self.apply_preconditioner(r, MhM_inv, matL)
            rz_new = torch.dot(r.conj().view(-1), z.view(-1)).real
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

            cnt += 1

        x = x.to(dtype_init)
        return x, cnt, errors[-1]
  

    def initialize_curl_mat(self):
        self.curl_mat_cpu = initialize_curl_mat(self.Lx, self.Ly).to(dtype)
        self.curl_mat = self.curl_mat_cpu.to(device=device)

    def initialize_geometry(self):
        Lx, Ly = self.Lx, self.Ly
        Vs = Lx * Ly
        # (x, y) in lattice
        xs = torch.arange(0, Lx, 2, device=device, dtype=torch.int64).unsqueeze(0)
        ys = torch.arange(0, Ly, 2, device=device, dtype=torch.int64).unsqueeze(1)
        # (i, j) in B_mat as in ci^dagger * cj, linearized with [Ly, Lx]
        self.i_list_1 = (xs + ys * Lx).view(-1)
        self.j_list_1 = ((xs + 1)%Lx + ys * Lx).view(-1)
        self.i_list_2 = self.i_list_1.clone()
        self.j_list_2 = (xs + (ys+1)%Ly * Lx).view(-1)
        self.i_list_3 = ((xs - 1)%Lx + ys * Lx).view(-1)
        self.j_list_3 = self.i_list_1.clone()
        self.i_list_4 = (xs + (ys-1)%Ly * Lx).view(-1)
        self.j_list_4 = self.i_list_1.clone()

        xs2 = torch.arange(1, Lx, 2, device=device, dtype=torch.int64).unsqueeze(0)
        ys2 = torch.arange(1, Ly, 2, device=device, dtype=torch.int64).unsqueeze(1)
        delta_i_list_1 = (xs2 + ys2 * Lx).view(-1)
        self.i_list_1 = torch.cat([self.i_list_1, delta_i_list_1])
        self.j_list_1 = torch.cat([self.j_list_1, ((xs2 + 1)%Lx + ys2 * Lx).view(-1)])
        self.i_list_2 = torch.cat([self.i_list_2, delta_i_list_1])
        self.j_list_2 = torch.cat([self.j_list_2, (xs2 + (ys2+1)%Ly * Lx).view(-1)])
        self.i_list_3 = torch.cat([self.i_list_3, ((xs2 - 1) % Lx + ys2 * Lx).view(-1)])
        self.j_list_3 = torch.cat([self.j_list_3, delta_i_list_1])
        self.i_list_4 = torch.cat([self.i_list_4, (xs2 + (ys2-1)%Ly * Lx).view(-1)])
        self.j_list_4 = torch.cat([self.j_list_4, delta_i_list_1])

        # The corresponding boson: [Ly, Lx, 2]
        self.boson_idx_list_1 = self.i_list_1 * 2
        self.boson_idx_list_2 = self.i_list_2 * 2 + 1
        self.boson_idx_list_3 = self.i_list_3 * 2
        self.boson_idx_list_4 = self.i_list_4 * 2 + 1

    def initialize_boson_test(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        self.boson = torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * torch.linspace(0.1, 0.5, self.bs, device=device).view(-1, 1, 1, 1, 1)


    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau, device=device)
        # self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.1

        self.boson = torch.randn(self.bs-1, 2, self.Lx, self.Ly, self.Ltau, device=device) * torch.linspace(0.1, 0.5, self.bs-1, device=device).view(-1, 1, 1, 1, 1)

        curl_mat = self.curl_mat * torch.pi/4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        delta_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])  
        self.boson = torch.cat([self.boson, delta_boson], dim=0)

    def initialize_boson_time_slice_random_normal(self):
        """
        Initialize bosons with random values within each time slice, keeping the values consistent across the spatial dimensions.

        :return: None
        """
        # Generate random values for each time slice
        random_values = torch.randn(self.bs-1, 2, self.Lx, self.Ly, device=device) * torch.linspace(0.01, 0.1, self.bs-1, device=device).view(-1, 1, 1, 1)

        # Repeat the random values across the time slices
        self.boson = random_values.unsqueeze(-1).repeat(1, 1, 1, 1, self.Ltau)

        curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        delta_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
        self.boson = torch.cat([self.boson, delta_boson], dim=0)

    def initialize_boson_time_slice_random_uniform(self):
        """
        Initialize bosons with random values within each time slice, keeping the values consistent across the spatial dimensions.

        :return: None
        """
        # Generate random values for each time slice using a uniform distribution in the range [-0.1, 0.1]
        random_values = (torch.rand(self.bs-1, 2, self.Lx, self.Ly, device=device) - 0.5) * 0.2

        # Repeat the random values across the time slices
        self.boson = random_values.unsqueeze(-1).repeat(1, 1, 1, 1, self.Ltau)

        curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        delta_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
        self.boson = torch.cat([self.boson, delta_boson], dim=0)


    def initialize_boson_staggered_pi(self):
        """
        Corresponding to self.i_list_1, i.e. the group_1 sites, the corresponding plaquettes have the right staggered pi pattern. This is directly obtainable from self.curl_mat

        "return boson: [bs, 2, Lx, Ly, Ltau]
        """
        curl_mat = self.curl_mat * torch.pi/4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        self.boson = boson.repeat(self.bs*self.Ltau, 1)
        self.boson = self.boson.reshape(self.bs, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])

    def apply_sigma_hat_cpu(self, i):
        if i % self.sigma_mini_batch_size == 0 and i > 0:
            self.sigma_hat = self.sigma_hat_cpu.to(self.sigma_hat.device)
            if mass_mode == -1:
                self.sigma_hat = self.sigma_hat ** (-1)

            # Normalize sigma_hat by its mean value
            self.sigma_hat = self.sigma_hat / self.sigma_hat.mean(dim=(2, 3, 4), keepdim=True)

            self.stabilize_sigma_hat(method='shrink', lmd=self.lmd)

            # # Find the 0.9 percentile value. Clip values in sigma_hat above the 0.9 percentile
            # reshaped_sigma_hat = self.sigma_hat.view(self.sigma_hat.size(0), self.sigma_hat.size(1), -1)
            # percentile_value = torch.quantile(reshaped_sigma_hat, 0.99, dim=-1, keepdim=True).view(self.sigma_hat.size(0), self.sigma_hat.size(1), 1, 1, 1)

            self.stabilize_sigma_hat(method='clamp', sgm_min=self.sig_min, sgm_max=self.sig_max)

            # # Flatten the sigma_hat tensor to 1D for plotting
            # sigma_hat_flat = self.sigma_hat.view(-1).cpu().numpy()
            # plt.figure(figsize=(10, 6))
            # plt.hist(sigma_hat_flat, bins=50, alpha=0.75, color='blue', edgecolor='black')
            # plt.title("Distribution of sigma_hat Values")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.grid(True)
            # plt.show()

            # max_values = self.sigma_hat.amax(dim=(2, 3, 4), keepdim=True)
            # self.sigma_hat = self.sigma_hat / (max_values)

            # sigma_hat_flat = self.sigma_hat.view(-1).cpu().numpy()
            # plt.figure(figsize=(10, 6))
            # plt.hist(sigma_hat_flat, bins=50, alpha=0.75, color='blue', edgecolor='black')
            # plt.title("Distribution of sigma_hat Values")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.grid(True)
            # plt.show()

            dbstop = 1

    def update_sigma_hat_cpu(self, boson_cpu, i):
        if i == self.annealing_step:
            self.sigma_hat_cpu = torch.ones(self.bs, 2, self.Lx, self.Ly, self.Ltau // 2 + 1, device='cpu', dtype=dtype)
            self.sigma_hat_mini_batch_last_i = i
        
        cnt = i - self.sigma_hat_mini_batch_last_i
        self.boson_mean_cpu = (self.boson_mean_cpu * cnt + boson_cpu) / (cnt + 1)
        delta_sigma_hat = torch.fft.rfftn(boson_cpu - self.boson_mean_cpu, dim=(2, 3, 4), norm="ortho").abs()**2
        self.sigma_hat_cpu = (self.sigma_hat_cpu * cnt + delta_sigma_hat) / (cnt + 1)
    
    def stabilize_sigma_hat(self, method="shrink", **kwargs):
        if method == "shrink":
            lmd = kwargs.get("lmd", 0.8)
            self.sigma_hat = (1-lmd) * self.sigma_hat + lmd * torch.ones_like(self.sigma_hat)
        elif method == "pow":
            beta = kwargs.get("beta", 0.5)
            self.sigma_hat = self.sigma_hat.pow(beta)
        elif method == "clamp":
            sgm_min = kwargs.get("sgm_min", 0.5)
            sgm_max = kwargs.get("sgm_max", 2.0)
            self.sigma_hat = self.sigma_hat.clamp(sgm_min, sgm_max)
        # add more strategies as needed
        # Exponential moving average
        # α = 0.05
        # σ_hat = (1-α) * σ_hat + α * delta_sigma_hat

    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [bs, 2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * math.sqrt(self.m)
    
    def draw_momentum_fft(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [bs, 2, Lx, Ly, Ltau] gaussian tensor
        """
        # return torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * math.sqrt(self.m)
        x = torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau // 2 + 1, device=device)
        y = torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau // 2 + 1, device=device)

        scale = torch.sqrt(self.sigma_hat * self.multiplier)
        z = (x + 1j * y) / scale
        return torch.fft.irfftn(z, (self.Lx, self.Ly, self.Ltau), norm="ortho")

    def apply_m_inv(self, p):
        """
        Apply the inverse of M to the momentum p. 
        p / self.m = M^{-1} @ p = Sigma @ p = Finv @ Sigma_hat @ F @ p
        :param p: [bs, 2, Lx, Ly, Ltau]
        :return: [bs, 2, Lx, Ly, Ltau] tensor
        """
        p_fft = torch.fft.rfftn(p, (self.Lx, self.Ly, self.Ltau))
        p_fft = p_fft * self.sigma_hat
        p = torch.fft.irfftn(p_fft, (self.Lx, self.Ly, self.Ltau))
        return p
 
    def draw_psudo_fermion(self):
        """
        Draw psudo_fermion psi = M(x0)'R
        :return: [bs, Ltau * Ly * Lx] gaussian tensor
        """
        R = torch.randn(self.bs, self.Lx * self.Ly * self.Ltau, device=device) / math.sqrt(2) + 1j * torch.randn(self.bs, self.Lx * self.Ly * self.Ltau, device=device) / math.sqrt(2)
        return R.to(dtype=cdtype)


    def sin_curl_greens_function_batch(self, boson):
        """
        Evaluate the Green's function of the curl of boson field.
        obsrv = curl(phi) at (x, y, tau1) * curl(phi) at (x, y, tau2),
        summed over (x, y) and averaged over the batch.

        :param boson: [batch_size, 2, Lx, Ly, Ltau] tensor
        :return: a tensor of shape [bs, num_dtau]
        """

        if boson is None:
            boson = self.boson

        if len(boson.shape) < 5:
            boson = boson.unsqueeze(0)
        
        Ltau = boson.shape[-1]

        # Compute the curl of boson
        phi_x = boson[:, 0]  # Shape: [batch_size, Lx, Ly, Ltau]
        phi_y = boson[:, 1]  # Shape: [batch_size, Lx, Ly, Ltau]

        sin_curl_phi = torch.sin(
            phi_x
            + torch.roll(phi_y, shifts=-1, dims=1)  # y-component at (x+1, y)
            - torch.roll(phi_x, shifts=-1, dims=2)  # x-component at (x, y+1)
            - phi_y
        )  # Shape: [batch_size, Lx, Ly, Ltau]

        correlations = []
        for dtau in range(self.num_tau + 1):
            idx1 = list(range(Ltau))
            idx2 = [(i + dtau) % Ltau for i in idx1]
            
            corr = torch.mean(sin_curl_phi[..., idx1] * sin_curl_phi[..., idx2], dim=(1, 2, 3))
            correlations.append(corr)

        return torch.stack(correlations).T  # Shape: [bs, num_dtau]


    def force_b_tau_cmp(self, boson):
        """
        x:  [bs, 2, Lx, Ly, Ltau]
        S = \sum (1 - cos(phi_tau+1 - phi))
        force_b_tau = -sin(phi_{tau+1} - phi_{tau}) + sin(phi_{tau} - phi_{tau-1})
        """    
        coeff = 1 / self.J / self.dtau**2
        diff_phi_tau_1 = torch.roll(boson, shifts=-1, dims=-1) - boson  # tau-component at (..., tau+1)
        diff_phi_tau_2 = boson - torch.roll(boson, shifts=1, dims=-1)  # tau-component at (..., tau-1)
        force_b_tau = -torch.sin(diff_phi_tau_1) + torch.sin(diff_phi_tau_2)
        return -coeff * force_b_tau       


    def action_boson_tau_cmp(self, x):
        """
        x:  [bs, 2, Lx, Ly, Ltau]
        S = \sum (1 - cos(phi_tau+1 - phi))
        """       
        coeff = 1 / self.J / self.dtau**2
        diff_phi_tau = - x + torch.roll(x, shifts=-1, dims=-1)  # tau-component at (..., tau+1)
        action = torch.sum(1 - torch.cos(diff_phi_tau), dim=(1, 2, 3, 4))
        return coeff * action
    
    def action_boson_plaq(self, boson):
        """
        boson: [bs, 2, Lx, Ly, Ltau]
        S: [bs,]
        """
        boson = boson.permute([3, 2, 1, 4, 0]).reshape(-1, self.Ltau, self.bs)
        curl = torch.einsum('ij,jkl->ikl', self.curl_mat_cpu if boson.device.type == 'cpu' else self.curl_mat, boson)  # [Vs, Ltau, bs]
        S = self.K * torch.sum(torch.cos(curl), dim=(0, 1))  
        return S

    def harmonic_tau(self, x0, p0, delta_t):
        """
        x:  [bs, 2, Lx, Ly, Ltau]
        p:  [bs, 2, Lx, Ly, Ltau]
        """
        v0 = p0 / self.m
        xk_0 = torch.fft.fft(x0, dim=-1)
        vk_0 = torch.fft.fft(v0, dim=-1)

        L = self.Ltau
        k = torch.fft.fftfreq(L)
        assert k[0].item() == 0

        # The unit is checked using `dtau * S_tau + dtau * p**2/(2m) + psi' (M'M)**(-1) psi`
        omega = torch.sqrt(1/self.m * 2 * (1 - torch.cos(k * 2*torch.pi)) / self.J / self.dtau**2).to(x0.device)

        # Evolve
        c = torch.cos(omega * delta_t)
        s = torch.sin(omega * delta_t)

        xk_t = torch.zeros_like(xk_0)
        vk_t = torch.zeros_like(vk_0)

        xk_t[..., 1:] = xk_0[..., 1:] * c[1:] + vk_0[..., 1:] / omega[1:] * s[1:]
        vk_t[..., 1:] = -omega[..., 1:] * xk_0[..., 1:] * s[1:] + vk_0[..., 1:] * c[1:]

        xk_t[..., 0] = xk_0[..., 0] + vk_0[..., 0]* delta_t;
        vk_t[..., 0] = vk_0[..., 0]

        # Transform back to position space
        xt = torch.fft.ifft(xk_t, dim=-1)
        xt = torch.real(xt)

        vt = torch.real(torch.fft.ifft(vk_t, dim=-1))
        pt = vt * self.m

        return xt, pt
 
    # =========== Turn on fermions =========
    def get_dB(self):
        Vs = self.Lx * self.Ly
        dB = torch.zeros(Vs, Vs, device=device, dtype=cdtype)
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        dB[diag_idx, diag_idx] = math.cosh(self.dtau/2 * self.t)
        return dB
    
    def get_dB_sparse(self):
        Vs = self.Lx * self.Ly
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        values = torch.full((Vs,), math.cosh(self.dtau / 2 * self.t), device=device, dtype=cdtype)
        indices = torch.stack([diag_idx, diag_idx])
        dB = torch.sparse_coo_tensor(indices, values, (Vs, Vs), device=device, dtype=cdtype).coalesce()
        return dB
    
    def get_dB_batch(self):
        Vs = self.Lx * self.Ly
        dB = torch.zeros(Vs, Vs, device=device, dtype=cdtype)
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        dB[diag_idx, diag_idx] = math.cosh(self.dtau/2 * self.t)
        return dB.repeat(self.bs, 1, 1)

    def get_dB1(self):
        Vs = self.Lx * self.Ly
        dB1 = torch.zeros(Vs, Vs, device=device, dtype=cdtype)
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        dB1[diag_idx, diag_idx] = math.cosh(self.dtau * self.t)
        return dB1
    
    def get_dB1_sparse(self):
        Vs = self.Lx * self.Ly
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        values = torch.full((Vs,), math.cosh(self.dtau * self.t), device=device, dtype=cdtype)
        indices = torch.stack([diag_idx, diag_idx])
        dB1 = torch.sparse_coo_tensor(indices, values, (Vs, Vs), device=device, dtype=cdtype).coalesce()
        return dB1

    def get_dB1_batch(self):
        Vs = self.Lx * self.Ly
        dB1 = torch.zeros(Vs, Vs, device=device, dtype=cdtype)
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        dB1[diag_idx, diag_idx] = math.cosh(self.dtau * self.t)
        return dB1.repeat(self.bs, 1, 1)
    
    def get_M(self, boson):
        """
        boson: [bs, 2, Lx, Ly, Ltau]
        """
        assert boson.size(0) == 1
        boson = boson.squeeze(0)

        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        boson = boson.permute([3, 2, 1, 0]).reshape(-1)

        M = torch.eye(Vs*self.Ltau, device=boson.device, dtype=cdtype)
        B1_list = []
        B2_list = []
        B3_list = []
        B4_list = []
        for tau in range(self.Ltau):
            dB1 = self.get_dB1()
            dB1[self.i_list_1, self.j_list_1] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1[self.j_list_1, self.i_list_1] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            B = dB1
            B1_list.append(dB1)

            dB = self.get_dB()
            dB[self.i_list_2, self.j_list_2] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB[self.j_list_2, self.i_list_2] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB
            B2_list.append(dB)

            dB = self.get_dB()
            dB[self.i_list_3, self.j_list_3] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB[self.j_list_3, self.i_list_3] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB
            B3_list.append(dB)

            dB = self.get_dB()
            dB[self.i_list_4, self.j_list_4] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB[self.j_list_4, self.i_list_4] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB
            B4_list.append(dB)

            if tau < self.Ltau - 1:
                row_start = Vs * (tau + 1)
                row_end = Vs * (tau + 2)
                col_start = Vs * tau
                col_end = Vs * (tau + 1)
                M[row_start:row_end, col_start:col_end] = -B
            else:
                M[:Vs, Vs*tau:] = B

        return M, [B1_list, B2_list, B3_list, B4_list]

    def get_diag_B_test(self, boson):
        """
        boson: [bs=1, 2, Lx, Ly, Ltau]
        """
        boson_input = boson
        assert len(boson.shape) == 4 or len(boson.shape) == 5 and boson.size(0) == 1
        if len(boson.shape) == 5:
            boson = boson.squeeze(0)

        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        boson = boson.permute([3, 2, 1, 0]).reshape(-1)

        mat = torch.sparse_coo_tensor(
            torch.arange(Vs * self.Ltau, device=boson.device).repeat(2, 1),
            torch.zeros(Vs * self.Ltau, dtype=cdtype, device=boson.device),
            (Vs * self.Ltau, Vs * self.Ltau),
            device=boson.device,
            dtype=cdtype
        )

        B1_list = []
        B2_list = []
        B3_list = []
        B4_list = []
        indices_list = []
        values_list = []

        for tau in range(self.Ltau):
            dB1 = self.get_dB1_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_1, self.j_list_1]),
                torch.stack([self.j_list_1, self.i_list_1])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_1]) * math.sinh(t * dtau),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            ])
            dB1 = torch.sparse_coo_tensor(
                torch.cat([dB1.indices(), indices], dim=1),
                torch.cat([dB1.values(), values]),
                dB1.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = dB1
            B1_list.append(dB1)

            dB = self.get_dB_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_2, self.j_list_2]),
                torch.stack([self.j_list_2, self.i_list_2])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_2]) * math.sinh(t * dtau / 2),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_2]) * math.sinh(t * dtau / 2)
            ])
            dB = torch.sparse_coo_tensor(
                torch.cat([dB.indices(), indices], dim=1),
                torch.cat([dB.values(), values]),
                dB.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = torch.sparse.mm(dB, torch.sparse.mm(B, dB))
            B2_list.append(dB)

            dB = self.get_dB_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_3, self.j_list_3]),
                torch.stack([self.j_list_3, self.i_list_3])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_3]) * math.sinh(t * dtau / 2),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_3]) * math.sinh(t * dtau / 2)
            ])
            dB = torch.sparse_coo_tensor(
                torch.cat([dB.indices(), indices], dim=1),
                torch.cat([dB.values(), values]),
                dB.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = torch.sparse.mm(dB, torch.sparse.mm(B, dB))
            B3_list.append(dB)

            dB = self.get_dB_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_4, self.j_list_4]),
                torch.stack([self.j_list_4, self.i_list_4])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_4]) * math.sinh(t * dtau / 2),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_4]) * math.sinh(t * dtau / 2)
            ])
            dB = torch.sparse_coo_tensor(
                torch.cat([dB.indices(), indices], dim=1),
                torch.cat([dB.values(), values]),
                dB.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = torch.sparse.mm(dB, torch.sparse.mm(B, dB))
            B4_list.append(dB)


            row_start = Vs * tau
            col_start = Vs * tau
            indices = B._indices() + torch.tensor([[row_start], [col_start]], device=device)
            values = B._values()

            indices_list.append(indices)
            values_list.append(values)

        # Combine all indices and values into M at once
        mat = torch.sparse_coo_tensor(
            torch.cat(indices_list, dim=1),
            torch.cat(values_list),
            mat.shape,
            device=device,
            dtype=mat.dtype
        ).coalesce()

        # MhM = torch.sparse.mm(M.T.conj(), M)  # Compute M'@M
        return mat


    def get_M_sparse(self, boson):
        """
        boson: [bs=1, 2, Lx, Ly, Ltau]
        """
        boson_input = boson
        assert len(boson.shape) == 4 or len(boson.shape) == 5 and boson.size(0) == 1, "Batch_Size > 1 is not supported"
        if len(boson.shape) == 5:
            boson = boson.squeeze(0)

        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        boson = boson.permute([3, 2, 1, 0]).reshape(-1)

        M = torch.sparse_coo_tensor(
            torch.arange(Vs * self.Ltau, device=boson.device).repeat(2, 1),
            torch.ones(Vs * self.Ltau, dtype=cdtype, device=boson.device),
            (Vs * self.Ltau, Vs * self.Ltau),
            device=boson.device,
            dtype=cdtype
        )
        
        B1_list = []
        B2_list = []
        B3_list = []
        B4_list = []
        indices_list = []
        values_list = []

        for tau in range(self.Ltau):
            dB1 = self.get_dB1_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_1, self.j_list_1]),
                torch.stack([self.j_list_1, self.i_list_1])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_1]) * math.sinh(t * dtau),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            ])
            dB1 = torch.sparse_coo_tensor(
                torch.cat([dB1.indices(), indices], dim=1),
                torch.cat([dB1.values(), values]),
                dB1.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = dB1
            B1_list.append(dB1)

            if tau in [-1]:
                blk_sparsity = 1.0 - (B._nnz() / (B.size(0) * B.size(1)))
                print(f"Sparsity of B at tau={tau}: {blk_sparsity:.4f}")
                print(f"Non-zero elements of B at tau={tau}: {B._nnz()}, Ratio: {B._nnz() / self.Lx:.4f}")
                blk_ratio = dB1._nnz() / self.Lx
                print(f"Ratio of non-zero elements of dB at tau={tau}: {blk_ratio:.4f}")

            dB = self.get_dB_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_2, self.j_list_2]),
                torch.stack([self.j_list_2, self.i_list_2])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_2]) * math.sinh(t * dtau / 2),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_2]) * math.sinh(t * dtau / 2)
            ])
            dB = torch.sparse_coo_tensor(
                torch.cat([dB.indices(), indices], dim=1),
                torch.cat([dB.values(), values]),
                dB.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = torch.sparse.mm(dB, torch.sparse.mm(B, dB))
            B2_list.append(dB)

            if tau in [-1]:
                blk_sparsity = 1.0 - (B._nnz() / (B.size(0) * B.size(1)))
                print(f"Sparsity of B at tau={tau}: {blk_sparsity:.4f}")
                print(f"Non-zero elements of B at tau={tau}: {B._nnz()}, Ratio: {B._nnz() / self.Lx:.4f}")
                blk_ratio = dB._nnz() / self.Lx
                print(f"Ratio of non-zero elements of dB at tau={tau}: {blk_ratio:.4f}")

            dB = self.get_dB_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_3, self.j_list_3]),
                torch.stack([self.j_list_3, self.i_list_3])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_3]) * math.sinh(t * dtau / 2),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_3]) * math.sinh(t * dtau / 2)
            ])
            dB = torch.sparse_coo_tensor(
                torch.cat([dB.indices(), indices], dim=1),
                torch.cat([dB.values(), values]),
                dB.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = torch.sparse.mm(dB, torch.sparse.mm(B, dB))
            B3_list.append(dB)

            if tau in [-1]:
                blk_sparsity = 1.0 - (B._nnz() / (B.size(0) * B.size(1)))
                print(f"Sparsity of B at tau={tau}: {blk_sparsity:.4f}")
                print(f"Non-zero elements of B at tau={tau}: {B._nnz()}, Ratio: {B._nnz() / self.Lx:.4f}")
                blk_ratio = dB._nnz() / self.Lx
                print(f"Ratio of non-zero elements of dB at tau={tau}: {blk_ratio:.4f}")

            dB = self.get_dB_sparse()
            indices = torch.cat([
                torch.stack([self.i_list_4, self.j_list_4]),
                torch.stack([self.j_list_4, self.i_list_4])
            ], dim=1)
            values = torch.cat([
                torch.exp(1j * boson[2 * Vs * tau + self.boson_idx_list_4]) * math.sinh(t * dtau / 2),
                torch.exp(-1j * boson[2 * Vs * tau + self.boson_idx_list_4]) * math.sinh(t * dtau / 2)
            ])
            dB = torch.sparse_coo_tensor(
                torch.cat([dB.indices(), indices], dim=1),
                torch.cat([dB.values(), values]),
                dB.shape,
                device=boson.device,
                dtype=cdtype
            ).coalesce()
            B = torch.sparse.mm(dB, torch.sparse.mm(B, dB))
            B4_list.append(dB)

            if tau in [-1]:
                blk_sparsity = 1.0 - (B._nnz() / (B.size(0) * B.size(1)))
                print(f"Sparsity of B at tau={tau}: {blk_sparsity:.4f}")
                print(f"Non-zero elements of B at tau={tau}: {B._nnz()}, Ratio: {B._nnz() / self.Lx:.4f}")
                blk_ratio = dB._nnz() / self.Lx
                print(f"Ratio of non-zero elements of dB at tau={tau}: {blk_ratio:.4f}")
            
                print('----------------')

            blk_sparsity = -1.0
            if tau in [-1]:
                blk_sparsity = 1.0 - (B._nnz() / (B.size(0) * B.size(1)))
                print(f"Sparsity of B at tau={tau}: {blk_sparsity:.4f}")
                blk_sparsity_BtB = 1.0 - (torch.sparse.mm(B.T.conj(), B)._nnz() / (B.size(0) * B.size(1)))
                print(f"Sparsity of B.T.conj() @ B at tau={tau}: {blk_sparsity_BtB:.4f}")

            if tau < self.Ltau - 1:
                row_start = Vs * (tau + 1)
                col_start = Vs * tau
                indices = B._indices() + torch.tensor([[row_start], [col_start]], device=device)
                values = -B._values()
            else:
                indices = B._indices() + torch.tensor([[0], [Vs * tau]], device=device)
                values = B._values()

            indices_list.append(indices)
            values_list.append(values)

        # Combine all indices and values into M at once
        M = torch.sparse_coo_tensor(
            torch.cat([M._indices()] + indices_list, dim=1),
            torch.cat([M._values()] + values_list),
            M.shape,
            device=device,
            dtype=M.dtype
        ).coalesce()

        MhM = torch.sparse.mm(M.T.conj(), M)  # Compute M'@M
        return MhM, [B1_list, B2_list, B3_list, B4_list], blk_sparsity, M

    def get_M_batch(self, boson):
        """
        boson: [bs, 2, Lx, Ly, Ltau]
        """
        # assert boson.size(0) == 1
        # boson = boson.squeeze(0)

        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        boson = boson.permute([0, 4, 3, 2, 1]).reshape(self.bs, -1)

        M = torch.eye(Vs * self.Ltau, device=boson.device, dtype=cdtype).repeat(self.bs, 1, 1)
        B1_list = []
        B2_list = []
        B3_list = []
        B4_list = []
        for tau in range(self.Ltau):
            dB1 = self.get_dB1_batch()
            dB1[:, self.i_list_1, self.j_list_1] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1[:, self.j_list_1, self.i_list_1] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            B = dB1
            B1_list.append(dB1)

            dB = self.get_dB_batch()
            dB[:, self.i_list_2, self.j_list_2] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB[:, self.j_list_2, self.i_list_2] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            B = torch.einsum('bij,bjk,bkl->bil', dB, B, dB)
            B2_list.append(dB)

            dB = self.get_dB_batch()
            dB[:, self.i_list_3, self.j_list_3] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB[:, self.j_list_3, self.i_list_3] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            B = torch.einsum('bij,bjk,bkl->bil', dB, B, dB)
            B3_list.append(dB)

            dB = self.get_dB_batch()
            dB[:, self.i_list_4, self.j_list_4] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB[:, self.j_list_4, self.i_list_4] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            B = torch.einsum('bij,bjk,bkl->bil', dB, B, dB)
            B4_list.append(dB)

            if tau < self.Ltau - 1:
                row_start = Vs * (tau + 1)
                row_end = Vs * (tau + 2)
                col_start = Vs * tau
                col_end = Vs * (tau + 1)
                M[:, row_start:row_end, col_start:col_end] = -B
            else:
                M[:, :Vs, Vs*tau:] = B

        return M, [B1_list, B2_list, B3_list, B4_list]

    def df_dot(self, boson_list, xi_lft, xi_rgt, i_list, j_list, is_group_1=False):
        """
        boson_list: [bs, Vs]
        xi_lft: [bs, Vs]
        xi_rgt: [bs, Vs]
        i_list: [Vs]
        j_list: [Vs]

        dB_{is, js}: (i, j, v), (j, i, v')
        """
        dtau = self.dtau
        t = self.t
        v = (math.sinh(t * dtau/2) if not is_group_1 else math.sinh(t * dtau)) * \
            torch.exp(1j * (boson_list + torch.pi/2))
        res = xi_lft[..., i_list] * xi_rgt[..., j_list] * v
        res += xi_lft[..., j_list] * xi_rgt[..., i_list] * v.conj()
        return res
    
    def df_dot_bs1(self, boson_list, xi_lft, xi_rgt, i_list, j_list, is_group_1=False):
        """
        boson_list: [bs, Vs]
        xi_lft: [bs, Vs]
        xi_rgt: [bs, Vs]
        i_list: [Vs]
        j_list: [Vs]

        dB_{is, js}: (i, j, v), (j, i, v')
        """
        xi_lft = xi_lft.to_dense().view(-1)
        xi_rgt = xi_rgt.to_dense().view(-1)

        dtau = self.dtau
        t = self.t
        v = (math.sinh(t * dtau/2) if not is_group_1 else math.sinh(t * dtau)) * \
            torch.exp(1j * (boson_list + torch.pi/2))
        res = xi_lft[..., i_list] * xi_rgt[..., j_list] * v
        res += xi_lft[..., j_list] * xi_rgt[..., i_list] * v.conj()
        return res

 
    def force_f_batch(self, psi, Mt, boson, B_list):
        """
        Ff(t) = -xi(t)[M'*dM + dM'*M]xi(t)

        [M(xt)'M(xt)] xi(t) = psi

        :param boson: [bs, 2, Lx, Ly, Ltau]
        :param Mt: [bs, Vs*Ltau, Vs*Ltau]
        :param psi: [bs, Vs*Ltau]

        :return Ft: [bs=1, 2, Lx, Ly, Ltau]
        :return xi: [bs, Lx*Ly*Ltau]
        """
        # assert boson.size(0) == 1
        boson = boson.permute([0, 4, 3, 2, 1]).reshape(self.bs, -1)

        Ot = torch.einsum('bij,bjk->bik', Mt.permute(0, 2, 1).conj(), Mt)
        L = torch.linalg.cholesky(Ot)
        Ot_inv = torch.cholesky_inverse(L)

        B1_list = B_list[0]
        B2_list = B_list[1]
        B3_list = B_list[2]
        B4_list = B_list[3]

        xi_t = torch.einsum('bij,bj->bi', Ot_inv, psi)

        Lx, Ly, Ltau = self.Lx, self.Ly, self.Ltau
        xi_t = xi_t.view(self.bs, Ltau, Ly * Lx)

        Ft = torch.zeros(self.bs, Ltau, Ly * Lx * 2, device=device, dtype=dtype)
        for tau in range(self.Ltau):
            xi_c = xi_t[:, tau].view(self.bs, -1, 1)
            xi_n = xi_t[:, (tau + 1) % Ltau].view(self.bs, -1, 1)
            B_xi = xi_c
            mat_Bs = [B4_list[tau], B3_list[tau], B2_list[tau], B1_list[tau], B2_list[tau], B3_list[tau], B4_list[tau]]
            for mat_B in mat_Bs:
                B_xi = torch.einsum('bij,bjk->bik', mat_B, B_xi)
            B_xi = B_xi.permute(0, 2, 1).conj()

            xi_n_lft_5 = xi_n.permute(0, 2, 1).conj().view(self.bs, -1)
            xi_n_lft_4 = torch.einsum('bi,bij->bj', xi_n_lft_5, B4_list[tau])
            xi_n_lft_3 = torch.einsum('bi,bij->bj', xi_n_lft_4, B3_list[tau])
            xi_n_lft_2 = torch.einsum('bi,bij->bj', xi_n_lft_3, B2_list[tau])
            xi_n_lft_1 = torch.einsum('bi,bij->bj', xi_n_lft_2, B1_list[tau])
            xi_n_lft_0 = torch.einsum('bi,bij->bj', xi_n_lft_1, B2_list[tau])
            xi_n_lft_m1 = torch.einsum('bi,bij->bj', xi_n_lft_0, B3_list[tau])

            xi_c_rgt_5 = xi_c.view(self.bs, -1)
            xi_c_rgt_4 = torch.einsum('bij,bj->bi', B4_list[tau], xi_c_rgt_5)
            xi_c_rgt_3 = torch.einsum('bij,bj->bi', B3_list[tau], xi_c_rgt_4)
            xi_c_rgt_2 = torch.einsum('bij,bj->bi', B2_list[tau], xi_c_rgt_3)
            xi_c_rgt_1 = torch.einsum('bij,bj->bi', B1_list[tau], xi_c_rgt_2)
            xi_c_rgt_0 = torch.einsum('bij,bj->bi', B2_list[tau], xi_c_rgt_1)
            xi_c_rgt_m1 = torch.einsum('bij,bj->bi', B3_list[tau], xi_c_rgt_0)

            B_xi_5 = B_xi.view(self.bs, -1)
            B_xi_4 = torch.einsum('bi,bij->bj', B_xi_5, B4_list[tau])
            B_xi_3 = torch.einsum('bi,bij->bj', B_xi_4, B3_list[tau])
            B_xi_2 = torch.einsum('bi,bij->bj', B_xi_3, B2_list[tau])
            B_xi_1 = torch.einsum('bi,bij->bj', B_xi_2, B1_list[tau])
            B_xi_0 = torch.einsum('bi,bij->bj', B_xi_1, B2_list[tau])
            B_xi_m1 = torch.einsum('bi,bij->bj', B_xi_0, B3_list[tau])

            Vs = self.Lx * self.Ly
            sign_B = -1 if tau < Ltau - 1 else 1

            boson_list = boson[:, 2 * Vs * tau + self.boson_idx_list_1]
            Ft[:, tau, self.boson_idx_list_1] = 2 * torch.real(
                self.df_dot(boson_list, xi_n_lft_2, xi_c_rgt_2, self.i_list_1, self.j_list_1, is_group_1=True) * sign_B
                + self.df_dot(boson_list, B_xi_2, xi_c_rgt_2, self.i_list_1, self.j_list_1, is_group_1=True)
            )

            boson_list = boson[:, 2 * Vs * tau + self.boson_idx_list_2]
            Ft[:, tau, self.boson_idx_list_2] = 2 * torch.real(
                self.df_dot(boson_list, xi_n_lft_3, xi_c_rgt_1, self.i_list_2, self.j_list_2) * sign_B
                + self.df_dot(boson_list, xi_n_lft_1, xi_c_rgt_3, self.i_list_2, self.j_list_2) * sign_B
                + self.df_dot(boson_list, B_xi_3, xi_c_rgt_1, self.i_list_2, self.j_list_2)
                + self.df_dot(boson_list, B_xi_1, xi_c_rgt_3, self.i_list_2, self.j_list_2)
            )

            boson_list = boson[:, 2 * Vs * tau + self.boson_idx_list_3]
            Ft[:, tau, self.boson_idx_list_3] = 2 * torch.real(
                self.df_dot(boson_list, xi_n_lft_4, xi_c_rgt_0, self.i_list_3, self.j_list_3) * sign_B
                + self.df_dot(boson_list, xi_n_lft_0, xi_c_rgt_4, self.i_list_3, self.j_list_3) * sign_B
                + self.df_dot(boson_list, B_xi_4, xi_c_rgt_0, self.i_list_3, self.j_list_3)
                + self.df_dot(boson_list, B_xi_0, xi_c_rgt_4, self.i_list_3, self.j_list_3)
            )

            boson_list = boson[:, 2 * Vs * tau + self.boson_idx_list_4]
            Ft[:, tau, self.boson_idx_list_4] = 2 * torch.real(
                self.df_dot(boson_list, xi_n_lft_5, xi_c_rgt_m1, self.i_list_4, self.j_list_4) * sign_B
                + self.df_dot(boson_list, xi_n_lft_m1, xi_c_rgt_5, self.i_list_4, self.j_list_4) * sign_B
                + self.df_dot(boson_list, B_xi_5, xi_c_rgt_m1, self.i_list_4, self.j_list_4)
                + self.df_dot(boson_list, B_xi_m1, xi_c_rgt_5, self.i_list_4, self.j_list_4)
            )

        # Ft = -Ft, neg from derivative inverse cancels neg dS/dx
        Ft = Ft.view(self.bs, Ltau, Ly, Lx, 2).permute(0, 4, 3, 2, 1)
        xi_t = xi_t.view(self.bs, -1)
        return Ft, xi_t
    

    def Ot_inv_psi_fast(self, psi, boson, MhM):
        # xi_t = torch.einsum('bij,bj->bi', Ot_inv, psi)
        # Ot = MhM.to_dense()
        # L = torch.linalg.cholesky(Ot)
        # Ot_inv = torch.cholesky_inverse(L)
        # xi_t = torch.einsum('ij,jk->ik', Ot_inv, psi[0].view(-1, 1))
        
        # err = torch.norm(MhM @ xi_t - psi[0].view(-1, 1))
        # print(f"Error in solving Ot_inv_psi_normal: {err}")
        # # return xi_t.view(-1)

        axs = None
        if self.plt_cg:
            fg, axs = plt.subplots()
        Ot_inv_psi, cnt_tensor, r_err = self.preconditioned_cg_fast_test(boson, psi, rtol_tensor=self.cg_rtol_tensor, max_iter=self.max_iter, MhM_inv=self.precon, MhM=MhM, axs=axs)

        # err2 = torch.norm(MhM @ Ot_inv_psi.view(-1, 1) - psi[0].view(-1, 1))
        # print(f"Error in solving Ot_inv_psi_fast: {err2}")

        return Ot_inv_psi, cnt_tensor, r_err


    def Ot_inv_psi(self, psi, MhM):
        # # xi_t = torch.einsum('bij,bj->bi', Ot_inv, psi)
        # Ot = MhM.to_dense()
        # L = torch.linalg.cholesky(Ot)
        # Ot_inv = torch.cholesky_inverse(L)
        # xi_t = torch.einsum('ij,jk->ik', Ot_inv, psi)
        # # return xi_t.view(-1)

        axs = None
        if self.plt_cg:
            fg, axs = plt.subplots()
        Ot_inv_psi, cnt, r_err = self.preconditioned_cg(MhM, psi, rtol=self.cg_rtol, max_iter=self.max_iter, MhM_inv=self.precon, cg_dtype=cg_dtype, axs=axs)

        return Ot_inv_psi, torch.tensor([cnt], device=device, dtype=torch.long)


    def force_f_fast(self, psi, boson, rtol, MhM):
        """
        Ff(t) = -xi(t)[M'*dM + dM'*M]xi(t)

        [M(xt)'M(xt)] xi(t) = psi

        :param boson: [bs, 2, Lx, Ly, Ltau]
        :param Mt: [bs, Vs*Ltau, Vs*Ltau]
        :param psi: [bs, Vs*Ltau]

        :return Ft: [bs, 2, Lx, Ly, Ltau]
        :return xi: [bs, Lx*Ly*Ltau]
        """
        self.cg_rtol_tensor = rtol

        # assert len(boson.shape) == 4 or boson.size(0) == 1
        # if len(boson.shape) == 5:
        #     boson = boson.squeeze(0)
        boson = boson.permute([0, 4, 3, 2, 1]).reshape(self.bs, self.Ltau, -1)

        xi_t, cg_converge_iter, r_err = self.Ot_inv_psi_fast(psi, boson, MhM)  # [bs, Lx*Ly*Ltau]

        Lx, Ly, Ltau = self.Lx, self.Ly, self.Ltau
        xi_t = xi_t.view(self.bs, Ltau, Ly * Lx)

        Ft = torch.empty(self.bs, Ltau, Ly * Lx * 2, device=device, dtype=dtype)
        for b in range(self.bs):
            for tau in range(self.Ltau):
                boson_in = boson[b, tau]

                xi_c = xi_t[b, tau].view(-1) # col
                xi_n = xi_t[b, (tau + 1) % Ltau].view(-1) # col

                B_xi_5 = _C.b_vec_per_tau(boson_in, xi_c, Lx, self.dtau, False, *BLOCK_SIZE)

                xi_n_conj = xi_n.conj()   # row
                xi_n_lft_conj = _C.b_vec_per_tau(boson_in, xi_n_conj, Lx, self.dtau, True, *BLOCK_SIZE)

                xi_c_rgt = _C.b_vec_per_tau(boson_in, xi_c, Lx, self.dtau, True, *BLOCK_SIZE)

                B_xi_5_conj = B_xi_5.conj()  # row
                B_xi_conj = _C.b_vec_per_tau(boson_in, B_xi_5_conj, Lx, self.dtau, True, *BLOCK_SIZE)


                sign_B = -1 if tau < Ltau - 1 else 1

                boson_list = boson[b, tau, self.boson_idx_list_1]
                Ft[b, tau, self.boson_idx_list_1] = 2 * torch.real(
                    self.df_dot_bs1(boson_list, xi_n_lft_conj[2].conj(), xi_c_rgt[2], self.i_list_1, self.j_list_1, is_group_1=True) * sign_B
                    + self.df_dot_bs1(boson_list, B_xi_conj[2].conj(), xi_c_rgt[2], self.i_list_1, self.j_list_1, is_group_1=True)
                )

                boson_list = boson[b, tau, self.boson_idx_list_2]
                Ft[b, tau, self.boson_idx_list_2] = 2 * torch.real(
                    self.df_dot_bs1(boson_list, xi_n_lft_conj[1].conj(), xi_c_rgt[3], self.i_list_2, self.j_list_2) * sign_B
                    + self.df_dot_bs1(boson_list, xi_n_lft_conj[3].conj(), xi_c_rgt[1], self.i_list_2, self.j_list_2) * sign_B
                    + self.df_dot_bs1(boson_list, B_xi_conj[1].conj(), xi_c_rgt[3], self.i_list_2, self.j_list_2)
                    + self.df_dot_bs1(boson_list, B_xi_conj[3].conj(), xi_c_rgt[1], self.i_list_2, self.j_list_2)
                )

                boson_list = boson[b, tau, self.boson_idx_list_3]
                Ft[b, tau, self.boson_idx_list_3] = 2 * torch.real(
                    self.df_dot_bs1(boson_list, xi_n_lft_conj[0].conj(), xi_c_rgt[4], self.i_list_3, self.j_list_3) * sign_B
                    + self.df_dot_bs1(boson_list, xi_n_lft_conj[4].conj(), xi_c_rgt[0], self.i_list_3, self.j_list_3) * sign_B
                    + self.df_dot_bs1(boson_list, B_xi_conj[0].conj(), xi_c_rgt[4], self.i_list_3, self.j_list_3)
                    + self.df_dot_bs1(boson_list, B_xi_conj[4].conj(), xi_c_rgt[0], self.i_list_3, self.j_list_3)
                )

                boson_list = boson[b, tau, self.boson_idx_list_4]
                Ft[b, tau, self.boson_idx_list_4] = 2 * torch.real(
                    self.df_dot_bs1(boson_list, xi_n_conj, xi_c_rgt[5], self.i_list_4, self.j_list_4) * sign_B
                    + self.df_dot_bs1(boson_list, xi_n_lft_conj[5].conj(), xi_c, self.i_list_4, self.j_list_4) * sign_B
                    + self.df_dot_bs1(boson_list, B_xi_5_conj, xi_c_rgt[5], self.i_list_4, self.j_list_4)
                    + self.df_dot_bs1(boson_list, B_xi_conj[5].conj(), xi_c, self.i_list_4, self.j_list_4)
                )

        # Ft = -Ft, neg from derivative inverse cancels neg dS/dx
        Ft = Ft.view(self.bs, Ltau, Ly, Lx, 2).permute(0, 4, 3, 2, 1)

        return Ft, xi_t.view(self.bs, -1), cg_converge_iter, r_err


    def force_f_sparse(self, psi, MhM, boson, B_list):
        """
        Ff(t) = -xi(t)[M'*dM + dM'*M]xi(t)

        [M(xt)'M(xt)] xi(t) = psi

        :param boson: [bs, 2, Lx, Ly, Ltau]
        :param Mt: [bs, Vs*Ltau, Vs*Ltau]
        :param psi: [bs, Vs*Ltau]

        :return Ft: [bs=1, 2, Lx, Ly, Ltau]
        :return xi: [bs, Lx*Ly*Ltau]
        """
        assert len(boson.shape) == 4 or boson.size(0) == 1
        if len(boson.shape) == 5:
            boson = boson.squeeze(0)
        boson = boson.permute([3, 2, 1, 0]).reshape(-1)

        B1_list = B_list[0]
        B2_list = B_list[1]
        B3_list = B_list[2]
        B4_list = B_list[3]

        # xi_t = torch.einsum('bij,bj->bi', Ot_inv, psi)
        xi_t, cg_converge_iter = self.Ot_inv_psi(psi, MhM)

        Lx, Ly, Ltau = self.Lx, self.Ly, self.Ltau
        xi_t = xi_t.view(Ltau, Ly * Lx)

        Ft = torch.zeros(Ltau, Ly * Lx * 2, device=device, dtype=dtype)
        for tau in range(self.Ltau):
            xi_c = xi_t[tau].view(-1, 1) # col
            xi_n = xi_t[(tau + 1) % Ltau].view(-1, 1) # col
            
            B_xi = xi_c  # column
            mat_Bs = [B4_list[tau], B3_list[tau], B2_list[tau], B1_list[tau], B2_list[tau], B3_list[tau], B4_list[tau]]
            for mat_B in mat_Bs:
                B_xi = torch.sparse.mm(mat_B, B_xi)
            B_xi = B_xi.permute(1, 0).conj()  # row

            xi_n_lft_5 = xi_n.conj().view(1, -1) # row
            xi_n_lft_4 = torch.sparse.mm(xi_n_lft_5, B4_list[tau])
            xi_n_lft_3 = torch.sparse.mm(xi_n_lft_4, B3_list[tau])
            xi_n_lft_2 = torch.sparse.mm(xi_n_lft_3, B2_list[tau])
            xi_n_lft_1 = torch.sparse.mm(xi_n_lft_2, B1_list[tau])
            xi_n_lft_0 = torch.sparse.mm(xi_n_lft_1, B2_list[tau])
            xi_n_lft_m1 = torch.sparse.mm(xi_n_lft_0, B3_list[tau])

            xi_c_rgt_5 = xi_c.view(-1, 1) # col
            xi_c_rgt_4 = torch.sparse.mm(B4_list[tau], xi_c_rgt_5)
            xi_c_rgt_3 = torch.sparse.mm(B3_list[tau], xi_c_rgt_4)
            xi_c_rgt_2 = torch.sparse.mm(B2_list[tau], xi_c_rgt_3)
            xi_c_rgt_1 = torch.sparse.mm(B1_list[tau], xi_c_rgt_2)
            xi_c_rgt_0 = torch.sparse.mm(B2_list[tau], xi_c_rgt_1)
            xi_c_rgt_m1 = torch.sparse.mm(B3_list[tau], xi_c_rgt_0)

            B_xi_5 = B_xi.view(1, -1)  # row
            B_xi_4 = torch.sparse.mm(B_xi_5, B4_list[tau])
            B_xi_3 = torch.sparse.mm(B_xi_4, B3_list[tau])
            B_xi_2 = torch.sparse.mm(B_xi_3, B2_list[tau])
            B_xi_1 = torch.sparse.mm(B_xi_2, B1_list[tau])
            B_xi_0 = torch.sparse.mm(B_xi_1, B2_list[tau])
            B_xi_m1 = torch.sparse.mm(B_xi_0, B3_list[tau])

            Vs = self.Lx * self.Ly
            sign_B = -1 if tau < Ltau - 1 else 1

            boson_list = boson[2 * Vs * tau + self.boson_idx_list_1]
            Ft[tau, self.boson_idx_list_1] = 2 * torch.real(
                self.df_dot_bs1(boson_list, xi_n_lft_2, xi_c_rgt_2, self.i_list_1, self.j_list_1, is_group_1=True) * sign_B
                + self.df_dot_bs1(boson_list, B_xi_2, xi_c_rgt_2, self.i_list_1, self.j_list_1, is_group_1=True)
            )

            boson_list = boson[2 * Vs * tau + self.boson_idx_list_2]
            Ft[tau, self.boson_idx_list_2] = 2 * torch.real(
                self.df_dot_bs1(boson_list, xi_n_lft_3, xi_c_rgt_1, self.i_list_2, self.j_list_2) * sign_B
                + self.df_dot_bs1(boson_list, xi_n_lft_1, xi_c_rgt_3, self.i_list_2, self.j_list_2) * sign_B
                + self.df_dot_bs1(boson_list, B_xi_3, xi_c_rgt_1, self.i_list_2, self.j_list_2)
                + self.df_dot_bs1(boson_list, B_xi_1, xi_c_rgt_3, self.i_list_2, self.j_list_2)
            )

            boson_list = boson[2 * Vs * tau + self.boson_idx_list_3]
            Ft[tau, self.boson_idx_list_3] = 2 * torch.real(
                self.df_dot_bs1(boson_list, xi_n_lft_4, xi_c_rgt_0, self.i_list_3, self.j_list_3) * sign_B
                + self.df_dot_bs1(boson_list, xi_n_lft_0, xi_c_rgt_4, self.i_list_3, self.j_list_3) * sign_B
                + self.df_dot_bs1(boson_list, B_xi_4, xi_c_rgt_0, self.i_list_3, self.j_list_3)
                + self.df_dot_bs1(boson_list, B_xi_0, xi_c_rgt_4, self.i_list_3, self.j_list_3)
            )

            boson_list = boson[2 * Vs * tau + self.boson_idx_list_4]
            Ft[tau, self.boson_idx_list_4] = 2 * torch.real(
                self.df_dot_bs1(boson_list, xi_n_lft_5, xi_c_rgt_m1, self.i_list_4, self.j_list_4) * sign_B
                + self.df_dot_bs1(boson_list, xi_n_lft_m1, xi_c_rgt_5, self.i_list_4, self.j_list_4) * sign_B
                + self.df_dot_bs1(boson_list, B_xi_5, xi_c_rgt_m1, self.i_list_4, self.j_list_4)
                + self.df_dot_bs1(boson_list, B_xi_m1, xi_c_rgt_5, self.i_list_4, self.j_list_4)
            )

        # Ft = -Ft, neg from derivative inverse cancels neg dS/dx
        Ft = Ft.view(Ltau, Ly, Lx, 2).permute(3, 2, 1, 0)

        return Ft, xi_t.view(-1), cg_converge_iter

    def leapfrog_proposer5_cmptau(self):
        """          
        Initially (x0, p0, psi) that satisfies e^{-H}, H = p^2/(2m) + Sb(x0) + Sf(x0, psi). Then evolve to (xt, pt, psi). 
        Sf(t) = psi' * [M(xt)'M(xt)]^(-1) * psi := R'R at t=0
        - dS/dx_{r, tau} = F_fermion
        Sample R ~ N(0, 1/sqrt(2)) + i N(0, 1/sqrt(2)), 
            -> psi = M(x0)'R 
            -> Sf(t) = psi' * [M(xt)'M(xt)]^(-1)*psi := psi' * xi(t)
        [M(xt)'M(xt)] xi(t) = psi

        Ff(t) = -xi(t)[M'*dM + dM'*M]xi(t)
            -> ...

        # Primitive leapfrog
        x_0 = x
        p_{1/2} = p_0 + dt/2 * F(x_0)

        x_{n+1} = x_{n} + p_{n+1/2}/m dt
        p_{n+3/2} = p_{n+1/2} + F(x_{n+1}) dt 

        p_{N} = (p_{N+1/2} + p_{N-1/2}) /2
        """

        # p0 = self.draw_momentum()  # [bs, 2, Lx, Ly, Ltau] tensor
        p0 = self.draw_momentum_fft()  # [bs, 2, Lx, Ly, Ltau] tensor
        x0 = self.boson  # [bs, 2, Lx, Ly, Ltau] tensor
        p = p0
        x = x0

        R_u = self.draw_psudo_fermion().view(-1, 1)
        if not self.use_cuda_kernel:
            result = self.get_M_sparse(x)
            MhM0, B_list, M0 = result[0], result[1], result[-1]
            psi_u = torch.sparse.mm(M0.permute(1, 0).conj(), R_u)
            force_f_u, xi_t_u, cg_converge_iter = self.force_f_sparse(psi_u, MhM0, x, B_list)
            psi_u = psi_u.view(self.bs, -1)
            xi_t_u = xi_t_u.view(self.bs, -1)

            r_err = torch.full((self.bs,), self.cg_rtol, dtype=dtype, device=device)
        else:
            psi_u = _C.mh_vec(x.permute([0, 4, 3, 2, 1]).reshape(self.bs, -1), R_u.view(self.bs, -1), self.Lx, self.dtau, *BLOCK_SIZE)
            # torch.testing.assert_close(psi_u, psi_u_ref, atol=1e-3, rtol=1e-3)

            # Use CUDA graph if available
            if self.cuda_graph and self.max_iter in self.force_graph_runners:
                force_f_u, xi_t_u, r_err = self.force_graph_runners[self.max_iter](psi_u, x, self.cg_rtol_tensor)
                cg_converge_iter = torch.full((self.bs,), self.max_iter, dtype=dtype, device=device)
            else:
                force_f_u, xi_t_u, cg_converge_iter, r_err = self.force_f_fast(psi_u, x, self.cg_rtol_tensor, None)
            # torch.testing.assert_close(force_f_u_ref.unsqueeze(0), force_f_u, atol=1e-3, rtol=1e-3)

        Sf0_u = torch.einsum('br,br->b', psi_u.conj(), xi_t_u)
        # Sf0_u = torch.dot(psi_u.conj().view(-1), xi_t_u.view(-1))
        # torch.testing.assert_close(torch.imag(Sf0_u), torch.zeros_like(torch.imag(Sf0_u)), atol=5e-3, rtol=1e-5)
        Sf0_u = torch.real(Sf0_u)

        assert x.grad is None

        Sb0 = self.action_boson_tau_cmp(x0) + self.action_boson_plaq(x0)
        H0 = Sb0 + torch.sum(p0 ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        H0 += Sf0_u

        dt = self.delta_t_tensor.view(-1, 1, 1, 1, 1)

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Sb_plaq = self.action_boson_plaq(x)
            force_b_plaq = -torch.autograd.grad(
                Sb_plaq, 
                x, 
                grad_outputs=torch.ones_like(Sb_plaq),
                create_graph=False)[0]
 
        force_b_tau = self.force_b_tau_cmp(x)

        if self.debug_pde:
            # print(f"Sb_tau={self.action_boson_tau(x)}")
            # print(f"p**2={torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)}")

            b_idx = 0
            if len(Sf0_u.shape) < 1:
                Sf0_u = Sf0_u.view(-1)

            # Initialize plot
            fig, axs = plt.subplots(3, 2, figsize=(12, 7.5))  # Two rows, one column

            Hs = [H0[b_idx].item()]
            # Ss = [(Sf0_u + Sf0_d + Sb0)[b_idx].item()]
            Sbs = [Sb0[b_idx].item()]
            Sbs_integ = [Sb0[b_idx].item()]
            Sfs = [Sf0_u[b_idx].item()]
            Sfs_integ = [Sf0_u[b_idx].item()]
            force_bs = [torch.linalg.norm((force_b_plaq + force_b_tau).reshape(self.bs, -1), dim=1)[b_idx].item()]
            force_fs = [torch.linalg.norm((force_f_u).reshape(self.bs, -1), dim=1)[b_idx].item()]

            # Setup for 1st subplot (Hs)
            line_Hs, = axs[0, 0].plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
            axs[0, 0].set_ylabel('Hamiltonian (H)')
            axs[0, 0].set_xticks([])  # Remove x-axis ticks
            axs[0, 0].legend()
            axs[0, 0].grid()

            axs[0, 1].set_xticks([])  
            axs[2, 1].set_xticks([])  

            # Setup for 2nd subplot (Ss)
            # axs[1].set_title('Real-Time Evolution of S_s')
            line_Sbs, = axs[1, 0].plot(Sbs, marker='s', linestyle='-', color='r', label='Sbs')
            axs[1, 0].set_ylabel('$S_b$')
            axs[1, 0].set_xticks([])  # Remove x-axis ticks
            axs[1, 0].legend()
            axs[1, 0].grid()

            line_Sbs_integ, = axs[0, 1].plot(Sbs_integ, marker='s', linestyle='-', color='r', label='Sbs')
            axs[0, 1].set_ylabel('$S_b$')
            axs[0, 1].set_xticks([])  # Remove x-axis ticks
            axs[0, 1].legend()
            axs[0, 1].grid()

            line_Sfs, = axs[1, 1].plot(Sfs, marker='s', linestyle='-', color='r', label='Sfs')
            axs[1, 1].set_ylabel('$S_f$')
            axs[1, 1].set_xticks([])  # Remove x-axis ticks
            axs[1, 1].legend()
            axs[1, 1].grid()

            line_Sfs_integ, = axs[2, 1].plot(Sfs_integ, marker='s', linestyle='-', color='r', label='Sfs')
            axs[2, 1].set_ylabel('$S_f$')
            axs[2, 1].set_xticks([])  # Remove x-axis ticks
            axs[2, 1].legend()
            axs[2, 1].grid()

            # Setup for 3rd subplot (force)
            line_force_b, = axs[2, 0].plot(force_bs, marker='s', linestyle='-', color='b', label='force_b')
            line_force_f, = axs[2, 0].plot(force_fs, marker='s', linestyle='-', color='r', label='force_f')
            axs[2, 0].set_xlabel('Leapfrog Step')
            axs[2, 0].set_ylabel('forces_norm')
            axs[2, 0].legend()
            axs[2, 0].grid()

        # Multi-scale Leapfrog
        # H(x, p) = U1/2 + sum_m (U0/2M + K/M + U0/2M) + U1/2 

        # x, p = self.harmonic_tau(x.repeat(2, 1, 1, 1, 1), p.repeat(2, 1, 1, 1, 1), self.dtau)
        # x = x[:1]
        # p = p[:1]

        cg_converge_iters = [cg_converge_iter]
        cg_r_errs = [r_err]
        for leap in range(self.N_leapfrog):

            p = p + dt/2 * (force_f_u)

            # Update (p, x)
            x_last = x
            M = 5
            for _ in range(M):
                # p = p + force(x) * dt/2
                # x = x + velocity(p) * dt
                # p = p + force(x) * dt/2

                p = p + (force_b_plaq + force_b_tau) * dt/2/M
                # x_ref = x + p / self.m * dt/M  # v = p/m ~ 1 / sqrt(m); dt'= sqrt(m) dt 
                x = x + self.apply_m_inv(p) * dt/M # v = p/m ~ 1 / sqrt(m); dt'= sqrt(m) dt 
                # torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-5)

                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    Sb_plaq = self.action_boson_plaq(x)
                    force_b_plaq = -torch.autograd.grad(Sb_plaq, x,
                    grad_outputs=torch.ones_like(Sb_plaq),
                    create_graph=False)[0]
                    
                force_b_tau = self.force_b_tau_cmp(x)

                p = p + (force_b_plaq + force_b_tau) * dt/2/M

            if not self.use_cuda_kernel:
                result = self.get_M_sparse(x)
                MhM = result[0]
                B_list = result[1]
                force_f_u, xi_t_u, cg_converge_iter = self.force_f_sparse(psi_u, MhM, x, B_list)
                xi_t_u = xi_t_u.view(self.bs, -1)
                
                r_err = torch.full((self.bs,), self.cg_rtol, dtype=dtype, device=device)
            else:
                if self.cuda_graph and self.max_iter in self.force_graph_runners:
                    force_f_u, xi_t_u, r_err = self.force_graph_runners[self.max_iter](psi_u, x, self.cg_rtol_tensor)
                    cg_converge_iter = torch.full((self.bs,), self.max_iter, dtype=dtype, device=device)
                else:
                    force_f_u, xi_t_u, cg_converge_iter, r_err = self.force_f_fast(psi_u, x, self.cg_rtol_tensor, None)
                # torch.testing.assert_close(force_f_u_ref.unsqueeze(0), force_f_u, atol=1e-3, rtol=1e-3)
            p = p + dt/2 * (force_f_u)

            cg_converge_iters.append(cg_converge_iter)
            cg_r_errs.append(r_err)
            if self.debug_pde:
                Sf_u = torch.real(torch.einsum('bi,bi->b', psi_u.conj(), xi_t_u))
                # Sf_u = torch.dot(psi_u.conj().view(-1), xi_t_u.view(-1)).view(-1)
                Sf_u = torch.real(Sf_u)
                if len(Sf0_u.shape) < 1:
                    Sf0_u = Sf0_u.view(-1)

                Sb_t = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x)
                H_t = Sb_t + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
                H_t += Sf_u

                dSb = (-torch.einsum('bi,bi->b', (x - x_last).view(self.bs, -1), (force_b_plaq + force_b_tau).reshape(self.bs, -1)))[b_idx]
                dSf = (-torch.einsum('bi,bi->b', (x - x_last).view(self.bs, -1), (force_f_u).reshape(self.bs, -1)))[b_idx]

                torch.testing.assert_close(H0, H_t, atol=1e-1, rtol=5e-3)

                # Hd, Sd = self.action((p + p_last)/2, x)  # Append new H value
                Hs.append(H_t[b_idx].item())
                Sbs.append(Sb_t[b_idx].item())
                Sbs_integ.append(Sbs_integ[-1] + dSb.cpu())
                Sfs.append((Sf_u)[b_idx].item())
                Sfs_integ.append(Sfs_integ[-1] + dSf.cpu())
                force_bs.append(torch.linalg.norm((force_b_plaq + force_b_tau).reshape(self.bs, -1), dim=1)[b_idx].item())
                force_fs.append(torch.norm((force_f_u).reshape(self.bs, -1), dim=1)[b_idx].item())

                # Update data for both subplots
                line_Hs.set_data(range(len(Hs)), Hs)
                line_Sbs.set_data(range(len(Sbs)), Sbs)
                line_Sbs_integ.set_data(range(len(Sbs_integ)), Sbs_integ)
                line_Sfs.set_data(range(len(Sfs)), Sfs)
                line_Sfs_integ.set_data(range(len(Sfs_integ)), Sfs_integ)
                line_force_b.set_data(range(len(force_bs)), force_bs)
                line_force_f.set_data(range(len(force_fs)), force_fs)

                # Adjust limits dynamically
                axs[0, 0].relim()
                axs[0, 0].autoscale_view()
                amp = max(Hs) - min(Hs)
                axs[0, 0].set_title(f'dt={self.delta_t:.2f}, m={self.m}, atol={amp:.2g}, rtol={amp/sum(Hs)*len(Hs):.2g}, N={self.N_leapfrog}')

                axs[1, 0].relim()
                axs[1, 0].autoscale_view()
                amp = max(Sbs) - min(Sbs)
                axs[1, 0].set_title(f'dt={self.delta_t:.3f}, m={self.m}, atol={amp:.2g}, N={self.N_leapfrog}')

                axs[1, 1].relim()
                axs[1, 1].autoscale_view()
                amp = max(Sfs) - min(Sfs)
                axs[1, 1].set_title(f'dt={self.delta_t:.3f}, m={self.m}, atol={amp:.2g}, N={self.N_leapfrog}')

                axs[0, 1].relim()
                axs[0, 1].autoscale_view()
                amp = max(Sbs_integ) - min(Sbs_integ)
                axs[0, 1].set_title(f'dt={self.delta_t:.3f}, m={self.m}, atol={amp:.2g}, N={self.N_leapfrog}')

                axs[2, 1].relim()
                axs[2, 1].autoscale_view()
                amp = max(Sfs_integ) - min(Sfs_integ)
                axs[2, 1].set_title(f'dt={self.delta_t:.3f}, m={self.m}, atol={amp:.2g}, N={self.N_leapfrog}')

                axs[2, 0].relim()
                axs[2, 0].autoscale_view()
                axs[2, 0].set_title(f'mean_force_b={sum(force_bs)/len(force_bs):.2g}, mean_force_f={sum(force_fs)/len(force_fs):.2g}')

                plt.pause(0.1)   # Small delay to update the plot

                # --------- save_plot ---------
                if leap == self.N_leapfrog - 1:
                    class_name = __file__.split('/')[-1].replace('.py', '')
                    method_name = "fermion_couple"
                    save_dir = os.path.join(script_path, f"./figures/leapfrog")
                    os.makedirs(save_dir, exist_ok=True)
                    file_path = os.path.join(save_dir, f"{method_name}_{self.specifics}.pdf")
                    plt.savefig(file_path, format="pdf", bbox_inches="tight")
                    print(f"Figure saved at: {file_path}")

        # Final energies
        Sf_fin_u = torch.einsum('br,br->b', psi_u.conj(), xi_t_u)
        # Sf_fin_u = torch.dot(psi_u.conj().view(-1), xi_t_u.view(-1)).view(-1)
        # torch.testing.assert_close(torch.imag(Sf_fin_u), torch.zeros_like(torch.real(Sf_fin_u)), atol=5e-2, rtol=1e-4)
        Sf_fin_u = torch.real(Sf_fin_u)

        Sb_fin = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x) 
        H_fin = Sb_fin + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        H_fin += Sf_fin_u

        # torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=0.05)
        # torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=5e-3)

        cg_converge_iters = torch.stack(cg_converge_iters)  # [sub_seq, bs]
        cg_r_errs = torch.stack(cg_r_errs)  # [sub_seq, bs]
        return x, H0, H_fin, cg_converge_iters.float().mean(dim=0), cg_r_errs.float().mean(dim=0)

    def metropolis_update(self):
        """
        Perform one step of metropolis update. Update self.boson.

        Given the last boson (conditional on the past) and momentum (iid sampled), the join dist. is the desired one. Then, the leapfrog proposes new config and the metropolis update preserves the join dist. The marginal dist. of the config is always conditional on the past while the momentum is not. Kinetic + potential (action) is conserved in the Hamiltonian dynamics but the action is not.

        :return: None
        """
        boson_new, H_old, H_new, cg_converge_iter, cg_r_err = self.leapfrog_proposer5_cmptau()
        accp = torch.rand(self.bs, device=device) < torch.exp(H_old - H_new)
        if debug_mode:
            print(f"H_old, H_new, diff: \n{H_old}, \n{H_new}, \n{H_new - H_old}")
            print(f"unclipped threshold: {torch.exp(H_old - H_new).tolist()}")
            print(f'Accp?: {accp.tolist()}')
            relative_error = torch.abs(H_new - H_old) / torch.abs(H_old)
            print(f"Relative error: {relative_error}")
        
        self.boson[accp] = boson_new[accp]
        
        # Calculate thresholds per batch element and add to respective queues
        thresholds = torch.minimum(torch.exp(H_old - H_new), torch.ones_like(H_old))
        for b in range(self.bs):
            self.threshold_queue[b].append(thresholds[b].item())
            
        return self.boson, accp, cg_converge_iter, cg_r_err
    
    # @torch.inference_mode()
    @torch.no_grad()
    def measure(self):
        """
        boson: [2, Lx, Ly, Ltau]

        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        # Initialization
        self.G_list[-1] = self.sin_curl_greens_function_batch(self.boson)
        self.S_plaq_list[-1] = self.action_boson_plaq(self.boson)
        self.S_tau_list[-1] = self.action_boson_tau_cmp(self.boson)

        # Warm up measure
        self.reset_precon()
        if self.cuda_graph:
            self.initialize_force_graph()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measure
        # fig = plt.figure()
        cnt_stream_write = 0
        executor = initialize_executor()
    
        # Use a dictionary to keep track of futures
        futures = {}

        for i in tqdm(range(self.N_step)):
            boson, accp, cg_converge_iter, cg_r_err = self.metropolis_update()
            
            # self.threshold_queue.append(threshold)
            if mass_mode != 0:
                self.apply_sigma_hat_cpu(i)
            self.adjust_delta_t()

            # Define CPU computations to run asynchronously
            def async_cpu_computations(i, boson_cpu, accp_cpu, cg_converge_iter_cpu, cg_r_err_cpu, delta_t_cpu, cnt_stream_write):
                # Update metrics
                self.accp_list[i] = accp_cpu
                self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
                self.G_list[i] = \
                    accp_cpu.view(-1, 1) * self.sin_curl_greens_function_batch(boson_cpu) \
                    + (1 - accp_cpu.view(-1, 1).to(torch.float)) * self.G_list[i-1]
                self.S_plaq_list[i] = \
                    accp_cpu.view(-1) * self.action_boson_plaq(boson_cpu) \
                    + (1 - accp_cpu.view(-1).to(torch.float)) * self.S_plaq_list[i-1]
                self.S_tau_list[i] = \
                    accp_cpu.view(-1) * self.action_boson_tau_cmp(boson_cpu) \
                    + (1 - accp_cpu.view(-1).to(torch.float)) * self.S_tau_list[i-1]
                    
                self.cg_iter_list[i] = cg_converge_iter_cpu
                self.cg_r_err_list[i] = cg_r_err_cpu
                self.delta_t_list[i] = delta_t_cpu
                self.boson_seq_buffer[cnt_stream_write] = boson_cpu.view(self.bs, -1)
                self.update_sigma_hat_cpu(boson_cpu, i)                
                return i  # Return the step index for identification

            # Submit new task to the executor
            future = executor.submit(
                async_cpu_computations, 
                i, 
                boson.cpu() if boson.is_cuda else boson.clone(),  # Detach and clone tensors to avoid CUDA synchronization
                accp.cpu() if accp.is_cuda else accp.clone(), 
                cg_converge_iter.cpu() if cg_converge_iter.is_cuda else cg_converge_iter.clone(), 
                cg_r_err.cpu() if cg_r_err.is_cuda else cg_r_err.clone(), 
                self.delta_t_tensor.cpu() if self.delta_t_tensor.is_cuda else self.delta_t_tensor.clone(),
                cnt_stream_write
            )
            futures[i] = future
            
            # Clean up completed futures to maintain memory efficiency
            # Only keep the most recent futures to avoid memory buildup
            completed_futures = [idx for idx, fut in list(futures.items()) if fut.done()]
            for idx in completed_futures:
                # Get the result to raise exceptions if there were any
                try:
                    futures[idx].result()
                except Exception as e:
                    print(f"Error in async computation at step {idx}: {e}")
                del futures[idx]
                
            # If there are too many pending futures, wait for some to complete
            if len(futures) > 2:  # Adjust this number based on your system resources
                # Wait for the oldest future to complete
                oldest_idx = min(futures.keys())
                try:
                    futures[oldest_idx].result()
                except Exception as e:
                    print(f"Error in async computation at step {oldest_idx}: {e}")
                del futures[oldest_idx]

            self.step += 1
            self.cur_step += 1
            cnt_stream_write += 1

            # ================ stats =============== #
            # stream writing
            # if cnt_stream_write % self.stream_write_rate == 0:
            #     data_folder = script_path + "/check_points/hmc_check_point/"
            #     file_name = f"stream_ckpt_N_{self.specifics}_step_{self.N_step}"
            #     self.save_to_file(self.boson_seq[:cnt_stream_write].cpu(), data_folder, file_name)  

            #     cnt_stream_write = 0

            # print(f"-----------> {torch.cuda.is_available()}, {i % self.memory_check_rate}, {i}\n")
            # tmp_file_path = os.path.join(script_path, "tmp_memory_usage.txt")
            # with open(tmp_file_path, "a") as tmp_file:
            #     tmp_file.write(f"-----------> {torch.cuda.is_available()}, {i % self.memory_check_rate}, {i}\n")
                
            if torch.cuda.is_available() and i % self.memory_check_rate == 0:
                # Check memory usage
                mem_usage = torch.cuda.memory_allocated() / (1024 ** 2)
                max_mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"Memory usage at step {i}: {mem_usage:.2f} MB")
                print(f"Max memory usage: {max_mem_usage:.2f} MB")
                
                # Write memory usage to a temporary file
                # tmp_file_path = os.path.join(script_path, "tmp_memory_usage.txt")
                # with open(tmp_file_path, "a") as tmp_file:
                #     tmp_file.write(f"Step {i}: Memory usage: {mem_usage:.2f} MB, Max memory usage: {max_mem_usage:.2f} MB\n")

            # plotting
            if i % self.plt_rate == 0 and i > 0:
                plt.pause(0.1)
                plt.close()
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)

            # checkpointing
            if i % self.ckp_rate == 0 and i > 0:
                res = {'boson': boson,
                        'step': self.step,
                        'G_list': self.G_list.cpu(),
                        'S_plaq_list': self.S_plaq_list.cpu(),
                        'S_tau_list': self.S_tau_list.cpu(),
                        'cg_iter_list': self.cg_iter_list.cpu(),
                        'cg_r_err_list': self.cg_r_err_list.cpu(),
                        'delta_t_list': self.delta_t_list.cpu()}
                
                data_folder = script_path + "/check_points/hmc_check_point_aug/"
                file_name = f"ckpt_N_{self.specifics}_step_{self.step-1}"
                self.save_to_file(res, data_folder, file_name)  

   
        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'G_list': self.G_list.cpu(),
               'S_plaq_list': self.S_plaq_list.cpu(),
               'S_tau_list': self.S_tau_list.cpu(),
               'cg_iter_list': self.cg_iter_list.cpu(),
               'cg_r_err_list': self.cg_r_err_list.cpu(),
               'delta_t_list': self.delta_t_list}

        # Save to file
        data_folder = script_path + "/check_points/hmc_check_point_aug/"
        file_name = f"ckpt_N_{self.specifics}_step_{self.N_step}"
        self.save_to_file(res, data_folder, file_name)  

        # Save stream data
        data_folder = script_path + "/check_points/hmc_check_point_aug/"
        file_name = f"stream_ckpt_N_{self.specifics}_step_{self.N_step}"
        self.save_to_file(self.boson_seq_buffer[:cnt_stream_write].cpu(), data_folder, file_name)  

        return G_avg, G_std

    # ------- Save to file -------
    def save_to_file(self, res, data_folder, filename):
        os.makedirs(data_folder, exist_ok=True)
        filepath = os.path.join(data_folder, f"{filename}.pt")
        torch.save(res, filepath)
        print(f"Data saved to {filepath}")

    # ------- Visualization -------
    def total_monitoring(self):
        """
        Visualize obsrv and accp in subplots.
        """
        # plt.figure()
        fig, axes = plt.subplots(3, 2, figsize=(12, 7.5))
        
        start = start_total_monitor  # to prevent from being out of scale due to init out-liers
        seq_idx = np.arange(start, self.cur_step, 1)
        seq_idx_all = np.arange(self.cur_step)

        axes[1, 0].plot(self.accp_rate[seq_idx].cpu().numpy())
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Acceptance Rate")
        axes[1, 0].grid()

        idx = [0, self.num_tau // 2, -2]
        # for b in range(self.G_list.size(1)):
        axes[0, 0].plot(self.G_list[seq_idx, ..., idx[0]].mean(axis=1).cpu().numpy(), label=f'G[0]')
        axes[0, 0].plot(self.G_list[seq_idx, ..., idx[1]].mean(axis=1).cpu().numpy(), label=f'G[{self.num_tau // 2}]')
        axes[0, 0].plot(self.G_list[seq_idx, ..., idx[2]].mean(axis=1).cpu().numpy(), label=f'G[-2]')
        axes[0, 0].set_ylabel("Greens Function")
        axes[0, 0].set_title("Greens Function Over Steps")
        axes[0, 0].legend()

        axes[0, 1].plot(self.S_plaq_list[seq_idx].cpu().numpy(), 'o', label='S_plaq')
        # axes[0, 1].plot(self.S_tau_list[seq_idx].cpu().numpy(), '*', label='S_tau')
        axes[0, 1].set_ylabel("$S_{plaq}$")
        axes[0, 1].legend()
        axes[0, 1].grid()

        # axes[1, 1].plot(self.S_tau_list[seq_idx].cpu().numpy() + self.S_plaq_list[seq_idx].cpu().numpy(), '*', label='$S_{tau}$')
        axes[1, 1].plot(self.S_tau_list[seq_idx].cpu().numpy(), '*', label='$S_{tau}$')
        axes[1, 1].set_ylabel("$S_{tau}$")
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].legend()

        # CG_converge_iter
        if not self.cuda_graph:
            axes[2, 0].plot(self.cg_iter_list[seq_idx_all].cpu().numpy(), '*', label=f'rtol_{self.cg_rtol}')
            axes[2, 0].set_ylabel("CG converge iter")
            axes[2, 0].set_xlabel("Steps")
            axes[2, 0].legend()
            axes[2, 0].grid()
        else:
            axes[2, 0].plot(self.cg_r_err_list[seq_idx_all].cpu().numpy(), '*', label=f'rtol_{self.cg_rtol}')
            axes[2, 0].set_ylabel("CG rel err")
            axes[2, 0].set_xlabel("Steps")
            axes[2, 0].legend()
            axes[2, 0].grid()

        # delta_t_iter
        axes[2, 1].plot(self.delta_t_list[seq_idx_all].cpu().numpy(), '*', label=r'$\delta t$')
        axes[2, 1].set_ylabel(r"$\delta t$")
        axes[2, 1].set_xlabel("Steps")
        axes[2, 1].legend()
        axes[2, 1].grid()

        plt.tight_layout()
        # plt.show(block=False)

        class_name = __file__.split('/')[-1].replace('.py', '')
        method_name = "totol_monit"
        save_dir = os.path.join(script_path, f"./figures/{class_name}_aug")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.specifics}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")

def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), step=1000001, 
                                       specifics='', plot_anal=True):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize
    filename = script_path + f"/check_points/hmc_check_point_aug/ckpt_N_{specifics}_step_{step}.pt"

    # filename = "/Users/kx/Desktop/hmc/fignote/local_vs_hmc_check/stat_check2/hmc_sampler_batch_rndm_real_space/hmc_check_point/ckpt_N_hmc_6_Ltau_10_Nstp_10000_Jtau_0.5_K_1_dtau_0.1_step_10000.pt"

    res = torch.load(filename)
    print(f'Loaded: {filename}')

    G_list = res['G_list']  # [seq, bs, num_tau+1]
    step = res['step']
    x = np.array(list(range(G_list[0].size(-1))))

    start = start_load
    end = step
    sample_step = 1
    seq_idx = torch.arange(start, end, sample_step)
    # batch_idx = torch.tensor([0, 1, 2, 3, 4])
    batch_size = G_list.size(1)
    batch_idx = torch.arange(batch_size)

    # -------- Plot -------- #
    plt.figure()
    plt.errorbar(x, G_list[seq_idx][:, batch_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx][:, batch_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.numpy().size), linestyle='-', marker='o', label='G_avg', color='blue', lw=2)

    for bi in range(G_list.size(1)):
        plt.errorbar(x, G_list[seq_idx, bi].numpy().mean(axis=(0)), yerr=G_list[seq_idx, bi].numpy().std(axis=(0))/np.sqrt(seq_idx.numpy().size), label=f'bs_{bi}', linestyle='--', marker='.', lw=2)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={end - start}")
    plt.legend()

    # Save plot
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens"
    save_dir = os.path.join(script_path, f"./figures/{class_name}_aug")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


    # -------- Log plot -------- #
    plt.figure()
    plt.errorbar(x+1, G_list[seq_idx][:, batch_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx][:, batch_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.numpy().size), linestyle='', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx][:, batch_idx].numpy().mean(axis=(0, 1))

    if plot_anal:
        # Analytical data
        data_folder_anal = script_path + "/../../hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
        name = f"N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{Ltau}_num_{Ltau}"
        pkl_filepath = os.path.join(data_folder_anal, f"{name}.pkl")

        with open(pkl_filepath, 'r') as f:
            loaded_data_anal = json.load(f)

        G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1]

        # Scale G_mean_anal to match G_mean based on their first elements
        idx_ref = 1
        scale_factor = G_mean[idx_ref] / G_mean_anal[idx_ref]
        G_mean_anal *= scale_factor

        plt.plot(x+1, G_mean_anal, linestyle='--', marker='*', label='G_avg_anal', color='red', lw=2, ms=10)


    # Add labels and title
    plt.xlabel('X-axis label')
    plt.ylabel('log10(G) values')
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={end - start}")
    plt.legend()
  
    plt.xscale('log')
    plt.yscale('log')

    # Save_plot 
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens_loglog"
    save_dir = os.path.join(script_path, f"./figures/{class_name}_aug")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")

    # -------- Plot S_tau_list -------- #
    S_tau_list = res['S_tau_list']  # [seq, bs]
    x = np.arange(len(seq_idx))

    plt.figure()
    plt.errorbar(x, S_tau_list[seq_idx][:, batch_idx].numpy().mean(axis=(1)), linestyle='', marker='.', label='S_tau_avg', color=f'C{9}', alpha=0.8, lw=2)
    avg_value = S_tau_list[seq_idx].numpy().mean(axis=(0, 1))
    plt.axhline(y=avg_value, color=f'C{9}', linestyle='-')   

    for bi in range(S_tau_list.size(1)):
        plt.errorbar(x, S_tau_list[seq_idx, bi].numpy(), label=f'S_tau_bs_{bi}', linestyle='', marker='.', lw=2, alpha=0.3, color=f'C{bi}')
        avg_value = S_tau_list[seq_idx, bi].numpy().mean()
        plt.axhline(y=avg_value, color=f'C{bi}', linestyle='-')

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$S_{\tau}(\tau)$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={end - start}")
    plt.legend()

    # Save plot
    method_name = "S_tau"
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


    # -------- Plot S_plaq_list -------- #
    S_plaq_list = res['S_plaq_list']  # [seq, bs]
    x = np.arange(len(seq_idx))

    plt.figure()
    plt.errorbar(x, S_plaq_list[seq_idx][:, batch_idx].numpy().mean(axis=(1)), linestyle='', marker='.', label='S_plaq_avg', color=f'C{9}', alpha=0.8, lw=2)
    avg_value = S_plaq_list[seq_idx].numpy().mean(axis=(0, 1))
    plt.axhline(y=avg_value, color=f'C{9}', linestyle='-')   

    for bi in range(S_plaq_list.size(1)):
        plt.errorbar(x, S_plaq_list[seq_idx, bi].numpy(), label=f'S_plaq_bs_{bi}', linestyle='', marker='.', lw=2, alpha=0.3, color=f'C{bi}')
        avg_value = S_plaq_list[seq_idx, bi].numpy().mean()
        plt.axhline(y=avg_value, color=f'C{bi}', linestyle='-')

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$S_{plaq}(\tau)$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={end - start}")
    plt.legend()

    # Save plot
    method_name = "S_plaq"
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


if __name__ == '__main__':
    J = float(os.getenv("J", '1.0'))
    Nstep = int(os.getenv("Nstep", '6000'))
    Lx = int(os.getenv("L", '6'))
    # Ltau = int(os.getenv("Ltau", '10'))
    # print(f'J={J} \nNstep={Nstep}')
    asym = int(os.environ.get("asym", '4'))

    Ltau = asym*Lx * 10 # dtau=0.1
    # Ltau = 10 # dtau=0.1

    print(f'J={J} \nNstep={Nstep} \nLx={Lx} \nLtau={Ltau}')
    hmc = HmcSampler(Lx=Lx, Ltau=Ltau, J=J, Nstep=Nstep)

    # Measure
    G_avg, G_std = hmc.measure()

    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc.N_step, hmc.specifics, False)

    plt.show()

    exit()

