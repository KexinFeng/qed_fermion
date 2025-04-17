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
# matplotlib.use('MacOSX')
plt.ion()
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from matplotlib import rcParams
rcParams['figure.raise_window'] = False
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.coupling_mat3 import initialize_curl_mat
from qed_fermion.post_processors.load_write2file_convert import time_execution
from qed_fermion import _C
from qed_fermion.force_graph_runner import ForceGraphRunner

BLOCK_SIZE = (4, 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f"device: {device}")

dtype = torch.float32
cdtype = torch.complex64
cg_dtype = torch.complex64
print(f"dtype: {dtype}")
print(f"cdtype: {cdtype}")
print(f"cg_cdtype: {cg_dtype}")

start_total_monitor = 1000
start_load = 2000

class HmcSampler(object):
    def __init__(self, Lx=6, Ltau=10, J=0.5, Nstep=3000, config=None):
        # Dims
        self.Lx = Lx
        self.Ly = Lx
        self.Ltau = Ltau
        self.bs = 1
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

        self.boson = None

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y
        self.plt_rate = 1000
        self.ckp_rate = 2000
        self.plt_cg = False
        self.verbose_cg = False
        self.stream_write_rate = Nstep
        self.memory_check_rate = 100

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

        # boson_seq
        self.boson_seq_buffer = torch.zeros(self.stream_write_rate, 2*self.Lx*self.Ly*self.Ltau, device=device, dtype=dtype)

        # Leapfrog
        self.debug_pde = False
        self.m = 1/2 * 4 / scale * 0.05
        # self.m = 1/2

        # self.delta_t = 0.05
        self.delta_t = 0.1/4
        # self.delta_t = 0.1 # This will be too large and trigger H0,Hfin not equal, even though N_leapfrog is cut half to 3

        # self.N_leapfrog = 6 # already oscilates back
        self.N_leapfrog = 4

        # CG
        self.cg_rtol = 1e-7
        # self.cg_rtol = 1e-4
        self.max_iter = 1000
        # self.max_iter = 300
        self.precon = None
        
        # CUDA Graph for force_f_fast
        self.use_cuda_graph = False  # Disable CUDA graph support for now
        print(f'use_cuda_graph:{ self.use_cuda_graph}')
        self.force_graph_runner = None
        self.force_graph_memory_pool = None
        self._BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 8, 16]
        self.force_graph_runners = {}

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
        
        # Initialize preconditioner
        self.reset_precon()
        
        # Initialize CUDA graph for force_f_fast if available
        if self.use_cuda_graph:
            self.initialize_force_graph()
     
    def Ot_inv_psi_fast(self, psi, boson, MhM):
        axs = None
        if self.plt_cg:
            fg, axs = plt.subplots()
        Ot_inv_psi, cnt, r_err = self.preconditioned_cg_fast_test(boson, psi, rtol=self.cg_rtol, max_iter=self.max_iter, MhM_inv=self.precon, MhM=MhM, axs=axs)
        return Ot_inv_psi, cnt

    def initialize_force_graph(self):
        """Initialize CUDA graph for force_f_fast function."""
        if not self.use_cuda_graph:
            return
            
        print("Initializing CUDA graph for force_f_fast...")
        
        # Create dummy inputs for the largest batch size
        max_batch_size = max(self._BATCH_SIZES_TO_CAPTURE)
        
        dummy_psi = torch.zeros(max_batch_size, self.Lx * self.Ly * self.Ltau, 
                               dtype=cdtype, device=device)
        dummy_boson = torch.zeros(max_batch_size, 2, self.Lx, self.Ly, self.Ltau, 
                                 dtype=dtype, device=device)
        
        # Capture graphs for different batch sizes
        for batch_size in reversed(self._BATCH_SIZES_TO_CAPTURE):
            graph_runner = ForceGraphRunner(self)
            graph_memory_pool = self.force_graph_memory_pool
            
            # Capture the graph
            graph_memory_pool = graph_runner.capture(
                dummy_psi[:batch_size],
                dummy_boson[:batch_size],
                graph_memory_pool
            )
            
            # Store the graph runner and memory pool
            self.force_graph_runners[batch_size] = graph_runner
            self.force_graph_memory_pool = graph_memory_pool
            
        print(f"CUDA graph initialization complete for batch sizes: {self._BATCH_SIZES_TO_CAPTURE}")
           
    def Ot_inv_psi_fast(self, psi, boson, MhM):
        axs = None
        if self.plt_cg:
            fg, axs = plt.subplots()
        Ot_inv_psi, cnt, r_err = self.preconditioned_cg_fast_test(boson, psi, rtol=self.cg_rtol, max_iter=self.max_iter, MhM_inv=self.precon, MhM=MhM, axs=axs)
        return Ot_inv_psi, cnt

    def preconditioned_cg_fast_test(self, boson, b, MhM_inv=None, matL=None, rtol=1e-7, max_iter=400, b_idx=None, axs=None, cg_dtype=torch.complex64, MhM=None):
        """
        Solve M'M x = b using preconditioned conjugate gradient (CG) algorithm.

        :param boson: boson.permute([0, 4, 3, 2, 1]).view(self.bs, self.Ltau, -1)
        :param b: Right-hand side vector, [bs, Ltau*Ly*Lx]
        :param tol: Tolerance for convergence
        :param max_iter: Maximum number of iterations
        :return: Solution vector x, [bs, Ltau*Ly*Lx]
        """
        boson = boson.view(self.bs, -1)
        b = b.view(self.bs, -1)
        norm_b = torch.norm(b, dim=1)

        # Initialize variables
        x = torch.zeros_like(b).view(self.bs, -1)
        r = b.view(self.bs, -1) - _C.mhm_vec(boson, x, self.Lx, self.dtau, *BLOCK_SIZE)
        z = _C.precon_vec(r, self.precon_csr, self.Lx)

        p = z
        rz_old = torch.einsum('bj,bj->b', r.conj(), z).real

        residuals = []

        cnt = 0
        for i in range(max_iter):
            # Matrix-vector product with M'M
            # Op = torch.sparse.mm(MhM, p)
            Op = _C.mhm_vec(boson, p, self.Lx, self.dtau, *BLOCK_SIZE)
            
            alpha = rz_old / torch.einsum('bj,bj->b', p.conj(), Op).real
            x += alpha.unsqueeze(-1) * p
            r -= alpha.unsqueeze(-1) * Op

            # Compute and store the error (norm of the residual)
            error = torch.norm(r, dim=1) / norm_b
            residuals.append(error.item())

            z = _C.precon_vec(r, self.precon_csr, self.Lx)
            rz_new = torch.einsum('bj,bj->b', r.conj(), z).real
            beta = rz_new / rz_old
            p = z + beta.unsqueeze(-1) * p
            rz_old = rz_new

            cnt += 1

        return x, cnt, residuals[-1]
      
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

    def force_f_fast(self, psi, boson, MhM):
        """
        Ff(t) = -xi(t)[M'*dM + dM'*M]xi(t)

        [M(xt)'M(xt)] xi(t) = psi

        :param boson: [bs, 2, Lx, Ly, Ltau]
        :param Mt: [bs, Vs*Ltau, Vs*Ltau]
        :param psi: [bs, Vs*Ltau]

        :return Ft: [bs, 2, Lx, Ly, Ltau]
        :return xi: [bs, Lx*Ly*Ltau]
        """
        # Use CUDA graph if available and batch size is supported
        batch_size = psi.size(0)
        if self.use_cuda_graph and batch_size in self.force_graph_runners:
            return self.force_graph_runners[batch_size](psi, boson)
            
        # Fall back to regular execution if CUDA graph is not available or batch size is not supported
        boson = boson.permute([0, 4, 3, 2, 1]).reshape(self.bs, self.Ltau, -1)

        # xi_t = torch.einsum('bij,bj->bi', Ot_inv, psi)
        # xi_t, cg_converge_iter = self.Ot_inv_psi(psi, MhM)
        xi_t, cg_converge_iter = self.Ot_inv_psi_fast(psi, boson, MhM)  # [bs, Lx*Ly*Ltau]

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

        return Ft, xi_t.view(self.bs, -1), cg_converge_iter
