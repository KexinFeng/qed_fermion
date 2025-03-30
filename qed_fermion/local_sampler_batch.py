from collections import defaultdict
import json
import math
import numpy as np

import matplotlib

import matplotlib.pyplot as plt
plt.ion()

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.coupling_mat3 import initialize_coupling_mat3, initialize_curl_mat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")
dtype = torch.float32
cdtype = torch.complex64

# torch.set_default_dtype(dtype)

class LocalUpdateSampler(object):
    def __init__(self, J=0.5, Nstep=2e2, bs=5, plt_rate=500, config=None):
        # Dims
        self.Lx = 10
        self.Ly = 10
        self.Ltau = 100
        self.bs = bs
        self.Vs = self.Lx * self.Ly * self.Ltau

        # Couplings
        self.Nf = 2
        # J = 0.5  # 0.5, 1, 3
        # self.dtau = 2/(J*self.Nf)

        self.dtau = 0.1
        # Lagrangian * dtau
        self.J = J / self.dtau * self.Nf / 4
        self.K = 1 * self.dtau * self.Nf * 1/2
        self.t = 1

        self.boson = None
        self.boson_energy = None
        self.fermion_energy = None

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        self.plt_rate = (self.Vs * plt_rate) 
        self.ckp_rate = (self.Vs * 2000)

        # Statistics
        self.N_step = int(Nstep) * self.Lx * self.Ly * self.Ltau
        self.step = 0
        self.cur_step = 0
        self.start = 0

        self.G_list = torch.zeros(self.N_step, self.bs, self.num_tau + 1)
        self.accp_list = torch.zeros(self.N_step, self.bs, dtype=torch.bool)
        self.accp_rate = torch.zeros(self.N_step, self.bs)
        self.S_plaq_list = torch.zeros(self.N_step, self.bs)
        self.S_tau_list = torch.zeros(self.N_step, self.bs)
        self.detM_sign_list = torch.zeros(self.N_step, self.bs)

        # Local window
        ws_table = defaultdict(lambda : 0.15) 
        ws_table.update({0.5: 0.15, 1:0.2, 3: 0.25})
        self.w = ws_table[J] * torch.pi

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False

        # Initialization
        self.reset()
    
    def reset(self):
        self.initialize_curl_mat()
        self.initialize_geometry()
        self.initialize_specifics()
        # self.initialize_boson_time_slice_random_uniform()
        # self.initialize_boson_staggered_pi()
        self.initialize_boson_uniform()
        self.boson = self.boson.to(device=device, dtype=dtype)
        
    def initialize_curl_mat(self):
        self.curl_mat = initialize_curl_mat(self.Lx, self.Ly).to(device=device, dtype=dtype)
        print(f'dtype={dtype}')

    def initialize_specifics(self):
        self.specifics = f"local_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}bs_{self.bs}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf:.2g}_dtau_{self.dtau:.2g}"
    
    def get_specifics(self):
        return f"local_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}bs_{self.bs}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf:.2g}_dtau_{self.dtau:.2g}"

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
        Initialize bosons with random values within each time slice; across the time dimensions, the values are consistent.

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
        Initialize bosons with random values within each time slice; across the time dimensions, the values are consistent.

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


    def initialize_boson_uniform(self):
        """
        Generate random boson field with uniform randomness across all dimensions
        """
        bs = self.bs
        self.boson = (torch.rand(bs, 2, self.Lx, self.Ly, self.Ltau, device=device) - 0.5) * torch.linspace(0.5 * 3.14, 2 * 3.14, bs, device=device).reshape(-1, 1, 1, 1, 1)

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

    # =========== Turn on fermions =========
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

    def get_dB(self):
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
        return dB1.repeat(self.bs, 1, 1)
 
    def get_detM(self, boson):
        """
        boson: [bs, 2, Lx, Ly, Ltau]
        B = e^V4/2 ... e^V1/2 e^V1/2 ... e^V4/2
        return |I + BLtau B_Ltau-1 ... B1|
        """
        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        bs = self.bs
        boson = boson.permute([0, 3, 2, 1, 4]).reshape(bs, -1)

        prodB = torch.eye(Vs, device=boson.device, dtype=cdtype).repeat(bs, 1, 1)
        for tau in reversed(range(self.Ltau)):
            dB1 = self.get_dB1()
            dB1[:, self.i_list_1, self.j_list_1] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1[:, self.j_list_1, self.i_list_1] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            B = dB1

            dB = self.get_dB()
            dB[:, self.i_list_2, self.j_list_2] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB[:, self.j_list_2, self.i_list_2] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            B = torch.einsum('bij,bjk,bkl->bil', dB, B, dB)

            dB = self.get_dB()
            dB[:, self.i_list_3, self.j_list_3] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB[:, self.j_list_3, self.i_list_3] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            B = torch.einsum('bij,bjk,bkl->bil', dB, B, dB)

            dB = self.get_dB()
            dB[:, self.i_list_4, self.j_list_4] = \
            torch.exp(1j * boson[:, 2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB[:, self.j_list_4, self.i_list_4] = \
            torch.exp(-1j * boson[:, 2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            B = torch.einsum('bij,bjk,bkl->bil', dB, B, dB)

            prodB = torch.einsum('bij,bjk->bik', prodB, B)

        I = torch.eye(Vs, device=boson.device, dtype=cdtype).repeat(bs, 1, 1)
        detM = torch.det(I + prodB)
        return detM

    def local_u1_proposer(self):
        """
        boson: [bs, 2, Lx, Ly, Ltau]

        return boson_new, H_old, H_new
        """
        # ======= update x ======== #

        boson_new = self.boson.clone()

        win_size = self.w

        # Select tau index based on the current step
        idx_x, idx_y, idx_tau = torch.unravel_index(torch.tensor([self.step], device=boson_new.device), (self.Lx, self.Ly, self.Ltau))

        delta = (torch.rand_like(boson_new[..., 0, idx_x, idx_y, idx_tau]) - 0.5) * 2 * win_size # Uniform in [-w/2, w/2]
        boson_new[..., 0, idx_x, idx_y, idx_tau] += delta

        # Compute new energy
        boson_energy_new = self.action_boson_tau_cmp(boson_new) + self.action_boson_plaq(boson_new)
        detM = self.get_detM(boson_new)
        fermion_energy_new = -self.Nf * torch.log(torch.real(detM))

        # Metropolis_update
        H0 = self.boson_energy + self.fermion_energy
        H_new = boson_energy_new + fermion_energy_new
        accp = torch.rand(self.bs, device=device) < torch.exp(-(H_new - H0))
        # print(f"H_old, H_new, new-old: {self.boson_energy}, {boson_energy_new}, {boson_energy_new - self.boson_energy}")
        # print(f"threshold: {torch.exp(self.boson_energy - boson_energy_new).item()}")

        # print(f'Accp?: {accp.item()}')
        self.boson[accp] = boson_new[accp]
        self.boson_energy[accp] = boson_energy_new[accp]
        self.fermion_energy[accp] = fermion_energy_new[accp]

        # ======= update y ======== #
        boson_new = self.boson.clone()

        win_size = self.w

        # Select tau index based on the current step
        idx_x, idx_y, idx_tau = torch.unravel_index(torch.tensor([self.step], device=boson_new.device), (self.Lx, self.Ly, self.Ltau))

        delta = (torch.rand_like(boson_new[..., 1, idx_x, idx_y, idx_tau]) - 0.5) * 2 * win_size  # Uniform in [-w/2, w/2]
        boson_new[..., 1, idx_x, idx_y, idx_tau] += delta

        # Compute new energy
        boson_energy_new = self.action_boson_tau_cmp(boson_new) + self.action_boson_plaq(boson_new)
        detM = self.get_detM(boson_new)
        fermion_energy_new = -self.Nf * torch.log(torch.real(detM))

        # Metropolis_update
        H0 = self.boson_energy + self.fermion_energy
        H_new = boson_energy_new + fermion_energy_new
        accp = torch.rand(self.bs, device=device) < torch.exp(-(H_new - H0))

        self.boson[accp] = boson_new[accp]
        self.boson_energy[accp] = boson_energy_new[accp]
        self.fermion_energy[accp] = fermion_energy_new[accp]

        return self.boson, accp, torch.sign(torch.real(detM)) 
  
    # @torch.inference_mode()
    @torch.no_grad()
    def measure(self):
        """
        boson: [2, Lx, Ly, Ltau]

        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        # Initialization
        if self.step == 0:
            self.G_list[-1] = self.sin_curl_greens_function_batch(self.boson).cpu()
            self.S_tau_list[-1] = self.action_boson_tau_cmp(self.boson).cpu()
            self.S_plaq_list[-1] = self.action_boson_plaq(self.boson).cpu()
            self.boson_energy = self.action_boson_tau_cmp(self.boson) + self.action_boson_plaq(self.boson)
            detM = self.get_detM(self.boson)
            self.fermion_energy = -self.Nf * torch.log(torch.real(detM))
            self.detM_sign_list[-1] = torch.sign(torch.real(detM))

        # Measure
        data_folder = script_path + "/check_points/local_check_point/"
        start = self.step
        for i in tqdm(range(start, start + self.N_step)):
            boson, accp, detM_sign = self.local_u1_proposer()

            accp_cpu = accp.cpu()
            self.accp_list[i] = accp_cpu
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].float(), axis=0).cpu()
            self.G_list[i] = \
                accp_cpu.view(-1, 1) * self.sin_curl_greens_function_batch(boson).cpu() \
                + (1 - accp_cpu.view(-1, 1).float()) * self.G_list[i-1]
            self.S_tau_list[i] = \
                accp_cpu.view(-1) * self.action_boson_tau_cmp(boson).cpu() \
                + (1 - accp_cpu.view(-1).float()) * self.S_tau_list[i-1]
            self.S_plaq_list[i] = \
                accp_cpu.view(-1) * self.action_boson_plaq(boson).cpu() \
                + (1 - accp_cpu.view(-1).float()) * self.S_plaq_list[i-1]
            self.detM_sign_list[i] = \
                accp_cpu.view(-1) * detM_sign.cpu() \
                + (1 - accp_cpu.view(-1).float()) * self.detM_sign_list[i-1]

            # plotting
            if i % self.plt_rate == 0 and i > 0:
                plt.pause(0.1)
                plt.close()
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)

            # checkpointing
            if i % self.ckp_rate == 0 and i > 0:
                res = {'boson': self.boson.cpu(),
                       'boson_energy': self.boson_energy.cpu(),
                       'fermion_energy': self.fermion_energy.cpu(),
                       'step': self.step,
                       'G_list': self.G_list,
                       'S_plaq_list': self.S_plaq_list,
                       'S_tau_list': self.S_tau_list,
                       'accp_rate': self.accp_rate,
                       'accp_list': self.accp_list,
                       'detM_sign_list': self.detM_sign_list}
                
                file_name = f"ckpt_N_{self.specifics}_step_{self.step}"
                self.save_to_file(res, data_folder, file_name) 

            self.step += 1

        res = {'boson': self.boson.cpu(),
               'boson_energy': self.boson_energy.cpu(),
               'fermion_energy': self.fermion_energy.cpu(),
               'step': self.step,
               'G_list': self.G_list,
               'S_plaq_list': self.S_plaq_list,
               'S_tau_list': self.S_tau_list,
               'accp_rate': self.accp_rate,
               'accp_list': self.accp_list,
               'detM_sign_list': self.detM_sign_list}


        # Save to file
        file_name = f"ckpt_N_{self.specifics}_step_{self.step}"
        self.save_to_file(res, data_folder, file_name)           
        return
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.boson = checkpoint['boson'].to(device=device, dtype=dtype)
        self.boson_energy = checkpoint['boson_energy'].to(device=device, dtype=dtype)
        self.fermion_energy = checkpoint['fermion_energy'].to(device=device, dtype=dtype)
        self.step = checkpoint['step']
        self.start = self.step
        self.G_list = checkpoint['G_list']
        self.S_plaq_list = checkpoint['S_plaq_list']
        self.S_tau_list = checkpoint['S_tau_list']
        self.accp_list = checkpoint['accp_list']
        self.accp_rate = checkpoint['accp_rate']
        print(f"Checkpoint loaded from {checkpoint_path}")

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
        
        start = 50  # to prevent from being out of scale due to init out-liers
        Vs = self.Vs
        seq_idx = np.arange(start * Vs, self.step, Vs)
 
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

        # axes[1, 1].plot(self.S_tau_list[seq_idx].cpu().numpy() + self.S_plaq_list[seq_idx].cpu().numpy(), '*', label='$S_{tau} + S_{plaq}$')
        axes[1, 1].plot(self.S_tau_list[seq_idx].cpu().numpy(), '*', label='$S_{tau}$')
        axes[1, 1].set_ylabel("$S_{tau}$")
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].legend()
        
        axes[2, 0].plot(self.detM_sign_list[seq_idx].cpu().numpy(), '*', label='$sign(|M|)$')
        axes[2, 0].set_ylabel("$sign(|M|)$")
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



def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), step=1000001, specifics='', plot_anal=True):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize
    Vs = Lx * Ly * Ltau

    filename = script_path + f"/check_points/local_check_point/ckpt_N_{specifics}_step_{step}.pt"
    res = torch.load(filename)
    print(f'Loaded: {filename}')

    G_list = res['G_list']  # [seq, bs, num_tau+1]
    step = res['step']
    x = np.array(list(range(G_list[0].size(-1))))

    start = 1000
    end = step
    sample_step = Vs
    seq_idx = torch.arange(start * Vs, end, sample_step)
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

    J = float(os.getenv("J", '10'))
    Nstep = int(os.getenv("Nstep", '2000'))
    bs = int(os.getenv("bs", '5'))
    print(f'J={J} \nNstep={Nstep}')

    hmc = LocalUpdateSampler(J=J, Nstep=Nstep, bs=bs, plt_rate=200)

    # Measure
    hmc.measure()

    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc.N_step, hmc.specifics, False)

    plt.show(block=True)

    exit()

