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
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.coupling_mat3 import initialize_coupling_mat3, initialize_curl_mat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

dtype = torch.float32
cdtype = torch.complex64
cg_dtype = torch.complex64
print(f"dtype: {dtype}")
print(f"cdtype: {cdtype}")
print(f"cg_cdtype: {cg_dtype}")

start_total_monitor = 0
start_load = 2000

class HmcSampler(object):
    def __init__(self, Lx=6, Ltau=10, J=0.5, Nstep=3000, config=None):
        # Dims
        self.Lx = Lx
        self.Ly = Lx
        self.Ltau = Ltau
        self.bs = 1
        self.Vs = self.Lx * self.Ly * self.Ltau

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
        self.plt_rate = 5
        self.ckp_rate = 5000
        self.plt_cg = False
        self.verbose_cg = False
        self.stream_write_rate = Nstep

        # Statistics
        self.N_step = Nstep
        self.step = 0
        self.cur_step = 0

        self.G_list = torch.zeros(self.N_step, self.bs, self.num_tau + 1, device=device)
        self.accp_list = torch.zeros(self.N_step, self.bs, dtype=torch.bool, device=device)
        self.accp_rate = torch.zeros(self.N_step, self.bs, device=device)
        self.S_plaq_list = torch.zeros(self.N_step, self.bs, device=device)
        self.S_tau_list = torch.zeros(self.N_step, self.bs, device=device)
        self.Sf_list = torch.zeros(self.N_step, self.bs, device=device)
        self.H_list = torch.zeros(self.N_step, self.bs, device=device)

        self.cg_iter_list = torch.zeros(self.N_step, self.bs)

        # boson_seq
        self.boson_seq_buffer = torch.zeros(self.stream_write_rate, 2*self.Lx*self.Ly*self.Ltau, device=device, dtype=dtype)

        # Leapfrog
        self.debug_pde = False
        self.m = 1/2 * 4 / scale * 0.05
        # self.m = 1/2

        self.delta_t = 0.05
        # self.delta_t = 0.1 # This will be too large and trigger H0,Hfin not equal, even though N_leapfrog is cut half to 3

        # self.N_leapfrog = 6 # already oscilates back
        self.N_leapfrog = 4

        # CG
        self.cg_rtol = 1e-7
        self.max_iter = 1000
        self.precon = None

        # Debug
        torch.manual_seed(0)

        # Initialization
        self.reset()
    
    def reset(self):
        self.Vs = self.Lx * self.Ly * self.Ltau
        self.initialize_curl_mat()
        self.initialize_geometry()
        self.initialize_specifics()
        self.initialize_boson_time_slice_random_uniform()

        # self.initialize_boson()
        # # self.initialize_boson_staggered_pi()   
        # self.boson[:] = 0
        # self.boson[0, 0, 0, 0, 1] = 1.0

        # self.boson_seq_buffer[0] = self.convert(self.boson.squeeze(0))
        # data_folder = script_path + "/check_points/hmc_check_point/"
        # file_name = f"stream_ckpt_N_{self.specifics}_step_{self.N_step}"
        # self.save_to_file(self.boson_seq_buffer[:1].cpu(), data_folder, file_name) 

        # # debug
        # Stau = self.action_boson_tau_cmp(self.boson)/36 
        # Splaq = self.action_boson_plaq(self.boson)/36
        # data_folder = script_path + "/check_points/hmc_check_point/"
        # file_name = "ener"
        # results_dict = {"Stau": Stau.tolist(), "Splaq": Splaq.tolist()}
        # self.save_to_file(results_dict, data_folder, file_name)

        # # Save the results to a .csv file
        # df = pd.DataFrame([results_dict])
        # csv_file_path = data_folder + file_name + '.csv'
        # df.to_csv(csv_file_path, index=False)
        # print(f"-----> save to {csv_file_path}")

        # exit(0)

    def reset_precon(self):
        curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        pi_flux_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
        self.precon = self.get_precon_scipy(pi_flux_boson, output_scipy=False)


    def get_precon_scipy(self, pi_flux_boson, output_scipy=True):
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
        result_indices, result_values = eng.preconditioner(
            matlab_indices, matlab_values, M.size(0), M.size(1), nargout=2
        )
        eng.quit()

        # Convert MATLAB results directly to PyTorch tensors
        result_indices = torch.tensor(result_indices, dtype=torch.long, device=M.device).T - 1
        result_values = torch.tensor(result_values, dtype=M.dtype, device=M.device).view(-1)

        if output_scipy:
            # Create MhM_inv as a sparse_coo_tensor
            MhM_inv = sp.csr_matrix(
                (np.array(result_values.cpu()),
                (np.array(result_indices[0].cpu()), np.array(result_indices[1].cpu()))),
                shape=(M.size(0), M.size(1)),
                dtype=np.complex128 if M.dtype == torch.complex128 else np.complex64
            )
            return MhM_inv
        else:
            # Create MhM_inv as a sparse_coo_tensor
            MhM_inv = torch.sparse_coo_tensor(
                result_indices,
                result_values,
                (M.size(0), M.size(1)),
                dtype=M.dtype,
                device=M.device
            ).coalesce()
            return MhM_inv
    
    def preconditioned_cg_bs1(self, MhM, b, MhM_inv=None, matL=None, rtol=1e-8, max_iter=None, b_idx=None, axs=None, cg_dtype=torch.complex128):
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
        z = torch.sparse.mm(MhM_inv, r)
        p = z
        rz_old = torch.dot(r.conj().view(-1), z.view(-1)).real

        errors = []
        residuals = []

        if self.plt_cg and axs is not None:
            # Plot intermediate results
            line_res = axs.plot(residuals, marker='o', linestyle='-', label='no precon.' if MhM_inv is None and matL is None else 'precon.', color=f'C{0}' if MhM_inv is None and matL is None else f'C{1}')
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

            # Check for convergence
            if error < rtol:
                if self.verbose_cg:
                    print(f"Converged in {i+1} iterations.")
                break

            # Update the preconditioner
            z = torch.sparse.mm(MhM_inv, r)
            rz_new = torch.dot(r.conj().view(-1), z.view(-1)).real
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

            cnt += 1

        x = x.to(dtype_init)
        return x, cnt, errors[-1]
  

    def initialize_curl_mat(self):
        self.curl_mat = initialize_curl_mat(self.Lx, self.Ly).to(device=device, dtype=dtype)

    def initialize_specifics(self):      
        self.specifics = f"hmc_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_bs{self.bs}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf*2:.2g}_dtau_{self.dtau:.2g}"

    def get_specifics(self):
        return f"hmc_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_bs{self.bs}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf*2:.2g}_dtau_{self.dtau:.2g}"

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

    def initialize_boson_time_slice_random(self):
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

    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [bs, 2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * math.sqrt(self.m)

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
        curl = torch.einsum('ij,jkl->ikl', self.curl_mat, boson)  # [Vs, Ltau, bs]
        S = self.K * torch.sum(torch.cos(curl), dim=(0, 1))  
        return S  
    
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


    def get_M_sparse(self, boson):
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

    def force_f(self, psis, Mt, boson, B_list):
        """
        Ff(t) = -xi(t)[M'*dM + dM'*M]xi(t)

        [M(xt)'M(xt)] xi(t) = psi

        :param boson: [bs, 2, Lx, Ly, Ltau]
        :param Mt: [bs, Vs*Ltau, Vs*Ltau]
        :param psi: [bs, Vs*Ltau]

        :return Ft: [bs=1, 2, Lx, Ly, Ltau]
        :return xi: [bs, Lx*Ly*Ltau]
        """
        assert boson.size(0) == 1
        boson = boson.squeeze(0).permute([3, 2, 1, 0]).reshape(-1)
        psis = [psi.T for psi in psis]

        Ot = Mt.T.conj() @ Mt
        L = torch.linalg.cholesky(Ot)
        Ot_inv = torch.cholesky_inverse(L)

        F_ts = []
        xi_ts = []

        B1_list = B_list[0]
        B2_list = B_list[1]
        B3_list = B_list[2]
        B4_list = B_list[3]
        for psi in psis:
            xi_t = Ot_inv @ psi

            # Works ONLY for bs = 1 below
            Lx, Ly, Ltau = self.Lx, self.Ly, self.Ltau
            xi_t = xi_t.view(Ltau, Ly*Lx)

            # Compute force
            Ft = torch.zeros(Ltau, Ly*Lx*2, device=device, dtype=dtype)
            for tau in range(self.Ltau):
                # keep O(Vs) complexity
                xi_c = xi_t[tau].view(-1, 1)  # c: current
                xi_n = xi_t[(tau + 1) % Ltau].view(-1, 1)  # n: next
                B_xi = xi_c
                mat_Bs = [B4_list[tau], B3_list[tau], B2_list[tau], B1_list[tau], B2_list[tau], B3_list[tau], B4_list[tau]]
                for mat_B in mat_Bs:
                    B_xi = mat_B @ B_xi
                B_xi = B_xi.T.conj()

                xi_n_lft_5 = xi_n.T.conj().view(-1)
                xi_n_lft_4 = (xi_n_lft_5 @ B4_list[tau]).view(-1)
                xi_n_lft_3 = (xi_n_lft_4 @ B3_list[tau]).view(-1)
                xi_n_lft_2 = (xi_n_lft_3 @ B2_list[tau]).view(-1)
                xi_n_lft_1 = (xi_n_lft_2 @ B1_list[tau]).view(-1)
                xi_n_lft_0 = (xi_n_lft_1 @ B2_list[tau]).view(-1)
                xi_n_lft_m1 = (xi_n_lft_0 @ B3_list[tau]).view(-1)

                xi_c_rgt_5 = xi_c.view(-1)
                xi_c_rgt_4 = (B4_list[tau] @ xi_c_rgt_5).view(-1)
                xi_c_rgt_3 = (B3_list[tau] @ xi_c_rgt_4).view(-1)
                xi_c_rgt_2 = (B2_list[tau] @ xi_c_rgt_3).view(-1)
                xi_c_rgt_1 = (B1_list[tau] @ xi_c_rgt_2).view(-1)
                xi_c_rgt_0 = (B2_list[tau] @ xi_c_rgt_1).view(-1)
                xi_c_rgt_m1 = (B3_list[tau] @ xi_c_rgt_0).view(-1)

                B_xi_5 = B_xi.view(-1)
                B_xi_4 = (B_xi_5 @ B4_list[tau]).view(-1)
                B_xi_3 = (B_xi_4 @ B3_list[tau]).view(-1)
                B_xi_2 = (B_xi_3 @ B2_list[tau]).view(-1)
                B_xi_1 = (B_xi_2 @ B1_list[tau]).view(-1)
                B_xi_0 = (B_xi_1 @ B2_list[tau]).view(-1)
                B_xi_m1 = (B_xi_0 @ B3_list[tau]).view(-1)

                # Find force
                # F = real((BXi)' dB Xi') * 2 + real(X'j dB Xi) * 2
                Vs = self.Lx * self.Ly
                sign_B = -1 if tau < Ltau-1 else 1

                boson_list = boson[2*Vs*tau + self.boson_idx_list_1]
                Ft[tau, self.boson_idx_list_1] = 2 * torch.real(\
                self.df_dot(boson_list, xi_n_lft_2, xi_c_rgt_2, self.i_list_1, self.j_list_1, is_group_1=True) * sign_B \
                + self.df_dot(boson_list, B_xi_2, xi_c_rgt_2, self.i_list_1, self.j_list_1, is_group_1=True)
                )

                boson_list = boson[2*Vs*tau + self.boson_idx_list_2]
                Ft[tau, self.boson_idx_list_2] = 2 * torch.real(\
                self.df_dot(boson_list, xi_n_lft_3, xi_c_rgt_1, self.i_list_2, self.j_list_2) * sign_B
                + self.df_dot(boson_list, xi_n_lft_1, xi_c_rgt_3, self.i_list_2, self.j_list_2) * sign_B \
                + self.df_dot(boson_list, B_xi_3, xi_c_rgt_1, self.i_list_2, self.j_list_2)
                + self.df_dot(boson_list, B_xi_1, xi_c_rgt_3, self.i_list_2, self.j_list_2)
                )

                boson_list = boson[2*Vs*tau + self.boson_idx_list_3]
                Ft[tau, self.boson_idx_list_3] = 2 * torch.real(\
                self.df_dot(boson_list, xi_n_lft_4, xi_c_rgt_0, self.i_list_3, self.j_list_3) * sign_B
                + self.df_dot(boson_list, xi_n_lft_0, xi_c_rgt_4, self.i_list_3, self.j_list_3) * sign_B \
                + self.df_dot(boson_list, B_xi_4, xi_c_rgt_0, self.i_list_3, self.j_list_3)
                + self.df_dot(boson_list, B_xi_0, xi_c_rgt_4, self.i_list_3, self.j_list_3)
                )

                boson_list = boson[2*Vs*tau + self.boson_idx_list_4]
                Ft[tau, self.boson_idx_list_4] = 2 * torch.real(\
                self.df_dot(boson_list, xi_n_lft_5, xi_c_rgt_m1, self.i_list_4, self.j_list_4) * sign_B
                + self.df_dot(boson_list, xi_n_lft_m1, xi_c_rgt_5, self.i_list_4, self.j_list_4) * sign_B  \
                + self.df_dot(boson_list, B_xi_5, xi_c_rgt_m1, self.i_list_4, self.j_list_4)
                + self.df_dot(boson_list, B_xi_m1, xi_c_rgt_5, self.i_list_4, self.j_list_4)
                )

            # Ft = -Ft  # neg from derivative inverse cancels neg dS/dx
            Ft = Ft.view(Ltau, Ly, Lx, 2).permute([3, 2, 1, 0])

            xi_t = xi_t.view(self.bs, -1)

            F_ts.append(Ft)
            xi_ts.append(xi_t)

        return tuple(F_ts), tuple(xi_ts)

     
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
    

    def Ot_inv_psi(self, psi, MhM):
        # xi_t = torch.einsum('bij,bj->bi', Ot_inv, psi)
        # Ot = MhM.to_dense()
        # L = torch.linalg.cholesky(Ot)
        # Ot_inv = torch.cholesky_inverse(L)
        # xi_t = torch.einsum('ij,jk->ik', Ot_inv, psi)
        # return xi_t.view(-1)
              
        # # Convert MhM to a scipy sparse matrix
        # MhM_scipy_csr = sp.csr_matrix(
        #     (MhM.values().cpu().numpy(),
        #      (MhM.indices()[0].cpu().numpy(), MhM.indices()[1].cpu().numpy())),
        #     shape=MhM.shape,
        #     dtype=MhM.values().cpu().numpy().dtype
        # )
        # psi_scipy = psi.cpu().numpy()

        # precon = sp.csr_matrix(
        #     (self.precon.values().cpu().numpy(),
        #      (self.precon.indices()[0].cpu().numpy(), self.precon.indices()[1].cpu().numpy())),
        #     shape=self.precon.shape,
        #     dtype=self.precon.values().cpu().numpy().dtype
        # )
        # # Ot_inv_psi = sp.linalg.spsolve(MhM_scipy_csr, psi_scipy)
        # Ot_inv_psi = sp.linalg.cg(MhM_scipy_csr, psi_scipy, rtol=self.cg_rtol, M=precon, maxiter=24)[0]
        # Ot_inv_psi = torch.tensor(Ot_inv_psi, device=psi.device, dtype=psi.dtype)

        Ot_inv_psi, cnt, _ = self.preconditioned_cg_bs1(MhM, psi, rtol=self.cg_rtol, max_iter=self.max_iter, MhM_inv=self.precon, cg_dtype=cg_dtype)
        return Ot_inv_psi, cnt

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

            xi_n_lft_5 = xi_n.permute(1, 0).conj().view(1, -1) # row
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

     
    def leapfrog_proposer4_cmptau(self):
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

        p0 = self.draw_momentum()  # [bs, 2, Lx, Ly, Ltau] tensor
        x0 = self.boson  # [bs, 2, Lx, Ly, Ltau] tensor
        p = p0
        x = x0

        R_u = self.draw_psudo_fermion()
        M0, B_list = self.get_M_batch(x)
        psi_u = torch.einsum('brs,bs->br', M0.permute(0, 2, 1).conj(), R_u)

        force_f_u, xi_t_u = self.force_f_batch(psi_u, M0, x, B_list)

        Sf0_u = torch.einsum('bi,bi->b', psi_u.conj(), xi_t_u)
        torch.testing.assert_close(torch.imag(Sf0_u), torch.zeros_like(torch.imag(Sf0_u)), atol=5e-3, rtol=1e-5)
        Sf0_u = torch.real(Sf0_u)

        assert x.grad is None

        Sb0 = self.action_boson_tau_cmp(x0) + self.action_boson_plaq(x0)
        H0 = Sb0 + torch.sum(p0 ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        H0 += Sf0_u

        dt = self.delta_t

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

            # Initialize plot
            fig, axs = plt.subplots(3, 2, figsize=(12, 7.5))  # Two rows, one column

            Hs = [H0[b_idx].item()]
            # Ss = [(Sf0_u + Sb0)[b_idx].item()]
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
                x = x + p / self.m * dt/M
                
                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    Sb_plaq = self.action_boson_plaq(x)
                    force_b_plaq = -torch.autograd.grad(Sb_plaq, x,
                    grad_outputs=torch.ones_like(Sb_plaq),
                    create_graph=False)[0]
                    
                force_b_tau = self.force_b_tau_cmp(x)

                p = p + (force_b_plaq + force_b_tau) * dt/2/M

            Mt, B_list = self.get_M_batch(x)
            force_f_u, xi_t_u = self.force_f_batch(psi_u, Mt, x, B_list)
            p = p + dt/2 * (force_f_u)

            if self.debug_pde:
                Sf_u = torch.real(torch.einsum('bi,bi->b', psi_u.conj(), xi_t_u))
                Sb_t = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x)
                H_t = Sb_t + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
                H_t += Sf_u

                dSb = -torch.dot((x - x_last).view(-1), (force_b_plaq + force_b_tau).reshape(-1))
                dSf = -torch.dot((x - x_last).view(-1), (force_f_u).reshape(-1))

                torch.testing.assert_close(H0, H_t, atol=1e-1, rtol=5e-2)

                # Hd, Sd = self.action((p + p_last)/2, x)  # Append new H value
                Hs.append(H_t[b_idx].item())
                Sbs.append(Sb_t[b_idx].item())
                Sbs_integ.append(Sbs_integ[-1] + dSb)
                Sfs.append((Sf_u)[b_idx].item())
                Sfs_integ.append(Sfs_integ[-1] + dSf)
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
        torch.testing.assert_close(torch.imag(Sf_fin_u).view(-1).cpu(), torch.zeros_like(torch.real(Sf_fin_u)), atol=5e-3, rtol=1e-4)
        Sf_fin_u = torch.real(Sf_fin_u)

        Sb_fin = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x) 
        H_fin = Sb_fin + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        H_fin += Sf_fin_u

        # torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=0.05)
        torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=1e-3)

        return x, H0, H_fin
    
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

        p0 = self.draw_momentum()  # [bs, 2, Lx, Ly, Ltau] tensor
        x0 = self.boson  # [bs, 2, Lx, Ly, Ltau] tensor
        p = p0
        x = x0

        R_u = self.draw_psudo_fermion().view(-1, 1)
        result = self.get_M_sparse(x)
        MhM0, B_list, M0 = result[0], result[1], result[-1]
        psi_u = torch.sparse.mm(M0.permute(1, 0).conj(), R_u)

        force_f_u, xi_t_u, cg_converge_iter = self.force_f_sparse(psi_u, MhM0, x, B_list)

        Sf0_u = torch.dot(psi_u.conj().view(-1), xi_t_u.view(-1))
        torch.testing.assert_close(torch.imag(Sf0_u), torch.zeros_like(torch.imag(Sf0_u)), atol=5e-3, rtol=1e-5)
        Sf0_u = torch.real(Sf0_u)

        assert x.grad is None

        Sb0 = self.action_boson_tau_cmp(x0) + self.action_boson_plaq(x0)
        H0 = Sb0 + torch.sum(p0 ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        H0 += Sf0_u

        dt = self.delta_t

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

        cg_converge_iters = [cg_converge_iter]
        for leap in range(self.N_leapfrog):

            p = p + dt/2 * (force_f_u.unsqueeze(0))

            # Update (p, x)
            x_last = x
            M = 5
            for _ in range(M):
                # p = p + force(x) * dt/2
                # x = x + velocity(p) * dt
                # p = p + force(x) * dt/2

                p = p + (force_b_plaq + force_b_tau) * dt/2/M
                x = x + p / self.m * dt/M
                
                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    Sb_plaq = self.action_boson_plaq(x)
                    force_b_plaq = -torch.autograd.grad(Sb_plaq, x,
                    grad_outputs=torch.ones_like(Sb_plaq),
                    create_graph=False)[0]
                    
                force_b_tau = self.force_b_tau_cmp(x)

                p = p + (force_b_plaq + force_b_tau) * dt/2/M

            result = self.get_M_sparse(x)
            MhM = result[0]
            B_list = result[1]
            force_f_u, xi_t_u, cg_converge_iter = self.force_f_sparse(psi_u, MhM, x, B_list)
            p = p + dt/2 * (force_f_u.unsqueeze(0))

            cg_converge_iters.append(cg_converge_iter)
            if self.debug_pde:
                # Sf_u = torch.real(torch.einsum('bi,bi->b', psi_u.conj(), xi_t_u))
                Sf_u = torch.dot(psi_u.conj().view(-1), xi_t_u.view(-1)).view(-1)
                torch.testing.assert_close(torch.imag(Sf_u), torch.zeros_like(torch.real(Sf_u)), atol=5e-3, rtol=1e-4)
                Sf_u = torch.real(Sf_u)
                if len(Sf0_u.shape) < 1:
                    Sf0_u = Sf0_u.view(-1)

                Sb_t = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x)
                H_t = Sb_t + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
                H_t += Sf_u

                dSb = -torch.dot((x - x_last).view(-1), (force_b_plaq + force_b_tau).reshape(-1))
                dSf = -torch.dot((x - x_last).view(-1), (force_f_u).reshape(-1))

                torch.testing.assert_close(H0, H_t, atol=1e-1, rtol=5e-2)

                # Hd, Sd = self.action((p + p_last)/2, x)  # Append new H value
                Hs.append(H_t[b_idx].item())
                Sbs.append(Sb_t[b_idx].item())
                Sbs_integ.append(Sbs_integ[-1] + dSb)
                Sfs.append((Sf_u)[b_idx].item())
                Sfs_integ.append(Sfs_integ[-1] + dSf)
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
        # Sf_fin_u = torch.einsum('br,br->b', psi_u.conj(), xi_t_u)
        Sf_fin_u = torch.dot(psi_u.conj().view(-1), xi_t_u.view(-1)).view(-1)
        torch.testing.assert_close(torch.imag(Sf_fin_u), torch.zeros_like(torch.real(Sf_fin_u)), atol=5e-3, rtol=1e-4)
        Sf_fin_u = torch.real(Sf_fin_u)

        Sb_fin = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x) 
        H_fin = Sb_fin + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        H_fin += Sf_fin_u

        # torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=0.05)
        torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=1e-3)

        return x, H0, H_fin, sum(cg_converge_iters)/len(cg_converge_iters)

    def metropolis_update(self):
        """
        Perform one step of metropolis update. Update self.boson.

        Given the last boson (conditional on the past) and momentum (iid sampled), the join dist. is the desired one. Then, the leapfrog proposes new config and the metropolis update preserves the join dist. The marginal dist. of the config is always conditional on the past while the momentum is not. Kinetic + potential (action) is conserved in the Hamiltonian dynamics but the action is not.

        :return: None
        """
        boson_new, H_old, H_new, cg_converge_iter = self.leapfrog_proposer5_cmptau()

        # print(f"H_old, H_new, diff: {H_old}, {H_new}, {H_new - H_old}")
        # print(f"threshold: {torch.exp(H_old - H_new).item()}")

        accp = torch.rand(self.bs, device=device) < torch.exp(H_old - H_new)
        # print(f'Accp?: {accp.item()}')
        self.boson[accp] = boson_new[accp]
        return self.boson, accp, cg_converge_iter
    
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
        self.reset_precon()

        # Measure
        # fig = plt.figure()
        cnt_stream_write = 0
        for i in tqdm(range(self.N_step)):
            boson, accp, cg_converge_iter = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
            self.G_list[i] = \
                accp.view(-1, 1) * self.sin_curl_greens_function_batch(boson) \
              + (1 - accp.view(-1, 1).to(torch.float)) * self.G_list[i-1]
            self.S_plaq_list[i] = \
                accp.view(-1) * self.action_boson_plaq(boson) \
              + (1 - accp.view(-1).to(torch.float)) * self.S_plaq_list[i-1]
            self.S_tau_list[i] = \
                accp.view(-1) * self.action_boson_tau_cmp(boson) \
              + (1 - accp.view(-1).to(torch.float)) * self.S_tau_list[i-1]
            # self.Sf_list[i] = torch.where(accp, Sf_fin, Sf0)

            self.cg_iter_list[i] = cg_converge_iter

            self.boson_seq_buffer[cnt_stream_write] = boson.squeeze(0).flatten()

            self.step += 1
            self.cur_step += 1
            cnt_stream_write += 1

            # stream writing
            # if cnt_stream_write % self.stream_write_rate == 0:
            #     data_folder = script_path + "/check_points/hmc_check_point/"
            #     file_name = f"stream_ckpt_N_{self.specifics}_step_{self.N_step}"
            #     self.save_to_file(self.boson_seq[:cnt_stream_write].cpu(), data_folder, file_name)  

            #     cnt_stream_write = 0
            
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
                        'cg_iter_list': self.cg_iter_list}
                
                data_folder = script_path + "/check_points/hmc_check_point/"
                file_name = f"ckpt_N_{self.specifics}_step_{self.step-1}"
                self.save_to_file(res, data_folder, file_name)  

   
        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'G_list': self.G_list.cpu(),
               'S_plaq_list': self.S_plaq_list.cpu(),
               'S_tau_list': self.S_tau_list.cpu(),
               'cg_iter_list': self.cg_iter_list}

        # Save to file
        data_folder = script_path + "/check_points/hmc_check_point/"
        file_name = f"ckpt_N_{self.specifics}_step_{self.N_step}"
        self.save_to_file(res, data_folder, file_name)  

        # Save stream data
        data_folder = script_path + "/check_points/hmc_check_point/"
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

        axes[1, 0].plot(self.accp_rate[seq_idx].cpu().numpy())
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Acceptance Rate")

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

        # axes[1, 1].plot(self.S_tau_list[seq_idx].cpu().numpy() + self.S_plaq_list[seq_idx].cpu().numpy(), '*', label='$S_{tau}$')
        axes[1, 1].plot(self.S_tau_list[seq_idx].cpu().numpy(), '*', label='$S_{tau}$')
        axes[1, 1].set_ylabel("$S_{tau}$")
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].legend()

        # CG_converge_iter
        axes[2, 0].plot(self.cg_iter_list[seq_idx].cpu().numpy(), '*', label='$CG_conv_iter$')
        axes[2, 0].set_ylabel("$CG_converge_iter$")
        axes[2, 0].set_xlabel("Steps")
        axes[2, 0].legend()

        plt.tight_layout()
        # plt.show(block=False)

        class_name = __file__.split('/')[-1].replace('.py', '')
        method_name = "totol_monit"
        save_dir = os.path.join(script_path, f"./figures/{class_name}")
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
    filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{specifics}_step_{step}.pt"

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
    save_dir = os.path.join(script_path, f"./figures/{class_name}")
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
    save_dir = os.path.join(script_path, f"./figures/{class_name}")
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
    Lx = int(os.getenv("Lx", '10'))
    Ltau = int(os.getenv("Ltau", '40'))

    print(f'J={J} \nNstep={Nstep} \nLx={Lx} \nLtau={Ltau}')

    hmc = HmcSampler(Lx=Lx, Ltau=Ltau, J=J, Nstep=Nstep)

    # Measure
    G_avg, G_std = hmc.measure()

    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc.N_step, hmc.specifics, False)

    plt.show()

    exit()

