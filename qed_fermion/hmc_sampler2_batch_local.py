import json
import math
import numpy as np

import matplotlib
# matplotlib.use('MacOSX')

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

from qed_fermion.coupling_mat3 import initialize_coupling_mat3, initialize_curl_mat

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"device: {device}")
dtype = torch.float32

class LocalUpdateSampler(object):
    def __init__(self, J=0.5, Nstep=2e2, config=None):
        # Dims
        self.Lx = 6
        self.Ly = 6
        self.Ltau = 10
        self.bs = 1
        self.Vs = self.Lx * self.Ly * self.Ltau

        # Couplings
        self.Nf = 2
        # J = 20  # 0.5, 1, 3
        # self.dtau = 1/(J*self.Nf)

        self.dtau = 0.1
        scale = self.dtau  # used to apply dtau
        self.J = J / scale * self.Nf / 4
        self.K = 1 * scale * self.Nf
        self.t = 1

        self.boson = None
        self.boson_energy = None
        self.fermion_energy = None

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 0
        self.N_step = int(Nstep) * self.Lx * self.Ly * self.Ltau
        self.step = 0
        self.cur_step = 0
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.bs, self.num_tau + 1, device=device)
        self.accp_list = torch.zeros(self.N_step, self.bs, dtype=torch.bool, device=device)
        self.accp_rate = torch.zeros(self.N_step, self.bs, device=device)
        self.S_plaq_list = torch.zeros(self.N_therm_step + self.N_step, self.bs)
        self.S_tau_list = torch.zeros(self.N_therm_step + self.N_step, self.bs)
        self.Sf_list = torch.zeros(self.N_therm_step + self.N_step, self.bs)
        self.H_list = torch.zeros(self.N_therm_step + self.N_step, self.bs)

        # Local window
        self.w = 0.1 * torch.pi

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False

        # Initialization
        self.reset()
    
    def reset(self):
        self.initialize_curl_mat()
        self.initialize_geometry()
        self.initialize_specifics()
        self.initialize_boson_staggered_pi()

    def initialize_curl_mat(self):
        self.curl_mat = initialize_curl_mat(self.Lx, self.Ly).to(device)

    def initialize_specifics(self):
        self.specifics = f"local_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_dtau_{self.dtau}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf:.2g}_t_{self.t}_Nleap_{6}_dt_{0.02}"
    
    def get_specifics(self):
        return f"local_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_dtau_{self.dtau}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf:.2g}_t_{self.t}_Nleap_{6}_dt_{0.02}"

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

        # Buffer
        # self.M = torch.eye(Vs*self.Ltau, device=device, dtype=torch.complex64)

    def get_dB(self):
        Vs = self.Lx * self.Ly
        dB = torch.zeros(Vs, Vs, device=device, dtype=torch.complex64)
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        dB[diag_idx, diag_idx] = math.cosh(self.dtau/2 * self.t)
        return dB
    
    def get_dB1(self):
        Vs = self.Lx * self.Ly
        dB1 = torch.zeros(Vs, Vs, device=device, dtype=torch.complex64)
        diag_idx = torch.arange(Vs, device=device, dtype=torch.int64)
        dB1[diag_idx, diag_idx] = math.cosh(self.dtau * self.t)
        return dB1
    
    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau, device=device)
        # self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.1

        self.boson = torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * torch.linspace(0.5, 1, self.bs, device=device)

    def initialize_boson_staggered_pi(self):
        """
        Corresponding to self.i_list_1, i.e. the group_1 sites, the corresponding plaquettes have the right staggered pi pattern. This is directly obtainable from self.curl_mat

        "return boson: [bs, 2, Lx, Ly, Ltau]
        """
        curl_mat = self.curl_mat * torch.pi/4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        self.boson = boson.repeat(self.bs*self.Ltau, 1)
        self.boson = self.boson.reshape(self.bs, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])

    def local_u1_proposer(self):
        """
        boson: [bs, 2, Lx, Ly, Ltau]

        return boson_new, H_old, H_new
        """
        boson_new = self.boson.clone()

        win_size = self.w

        # Select tau index based on the current step
        idx_x, idx_y, idx_tau = torch.unravel_index(torch.tensor([self.step], device=boson_new.device), (self.Lx, self.Ly, self.Ltau))

        delta = (torch.rand_like(boson_new[..., idx_x, idx_y, idx_tau]) - 0.5) * 2 * win_size # Uniform in [-w/2, w/2]
        boson_new[..., idx_x, idx_y, idx_tau] += delta

        # return boson_new, (idx_x, idx_y, idx_tau)

        # Compute new energy
        boson_energy_new = self.action_boson_tau_cmp(boson_new) + self.action_boson_plaq(boson_new)
        detM = self.get_detM(boson_new)
        # detM_np = self.get_detM_np(boson_new)
        fermion_energy_new = -self.Nf * torch.log(torch.real(detM)).unsqueeze(0)

        # Metropolis_update
        accp = torch.rand(self.bs, device=device) < torch.exp(self.boson_energy + self.fermion_energy - boson_energy_new - fermion_energy_new)
        # print(f"H_old, H_new, new-old: {self.boson_energy}, {boson_energy_new}, {boson_energy_new - self.boson_energy}")
        # print(f"threshold: {torch.exp(self.boson_energy - boson_energy_new).item()}")

        # print(f'Accp?: {accp.item()}')
        fermion_energy_old = self.fermion_energy.clone()
        plaq_term_old = self.action_boson_plaq(self.boson)
        
        self.boson[accp] = boson_new[accp]
        self.boson_energy[accp] = boson_energy_new[accp]
        self.fermion_energy[accp] = fermion_energy_new[accp]
        
        energies_old = (fermion_energy_old, plaq_term_old, 0)
        energies_new = (fermion_energy_new, self.action_boson_plaq(self.boson), 0)
        return self.boson, accp, energies_old, energies_new


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
    
    # def action_boson_tau(self, x):
    #     """
    #     x:  [bs, 2, Lx, Ly, Ltau]

    #     S = 1/2 * x' @ A @ x
    #         F_{r,k} = e^{irk}
    #         F'F/L = FF'/L = i.d.
    #         x_k = fft(x) = F'@x = sum_r e^{-irk} x_r
    #         x_r = ifft(xk) = 1/L * F @ x_k = 1/L * sum_k e^{ikr} x_k

    #     S = 1/2 * x'@F @ F'@A@F @ F'@x / L**2
    #       = 1/2 * xk' @ F'F diag(A_k) @ xk / L**2
    #       = 1/2 * xk' @ diag(A_k) @ xk / L

    #     k = torch.fft.fftfreq(L) = [\kap/L] in the unit of 2*torch.pi
    #     A_k = 2 * (1 - torch.cos(k * 2*torch.pi))

    #     """
    #     coeff = 1/2 / self.J / self.dtau**2
    #     xk = torch.fft.fft(x, dim=-1)
    #     L = self.Ltau
    #     k = torch.fft.fftfreq(L).to(device)    
    #     lmd = 2 * (1 - torch.cos(k * 2*torch.pi))[None, None, None, None]

    #     # F = torch.fft.fft(torch.eye(3), dim=0)
    #     # torch.eye(3) = 1/L * F.conj().T @ F
    #     return coeff * (xk.abs()**2 * lmd).sum(dim=(1, 2, 3, 4)) / L

    def get_detM_np(self, boson):

        """
        boson: [bs, 2, Lx, Ly, Ltau]
        B = e^V4/2 ... e^V1/2 e^V1/2 ... e^V4/2
        return |I + BLtau B_Ltau-1 ... B1|
        """
        assert boson.size(0) == 1
        boson = boson.squeeze(0)

        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        boson = boson.permute([3, 2, 1, 0]).reshape(-1)
        
        prodB = np.eye(Vs, dtype=np.complex64)
        for tau in reversed(range(self.Ltau)):
            dB1 = self.get_dB1()
            dB1[self.i_list_1, self.j_list_1] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1[self.j_list_1, self.i_list_1] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1 = dB1.numpy()
            B = dB1
            # self.B1 = dB1

            dB = self.get_dB()
            dB[self.i_list_2, self.j_list_2] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB[self.j_list_2, self.i_list_2] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB = dB.numpy()
            B = dB @ B @ dB
            # self.B2 = dB

            dB = self.get_dB()
            dB[self.i_list_3, self.j_list_3] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB[self.j_list_3, self.i_list_3] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB = dB.numpy()
            B = dB @ B @ dB
            # self.B3 = dB

            dB = self.get_dB()
            dB[self.i_list_4, self.j_list_4] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB[self.j_list_4, self.i_list_4] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB = dB.numpy()
            B = dB @ B @ dB
            # self.B4 = dB   

            prodB = prodB @ B

        I = torch.eye(Vs, device=boson.device, dtype=torch.complex64).numpy()
        return np.linalg.det(I + prodB)


    def get_detM(self, boson):
        """
        boson: [bs, 2, Lx, Ly, Ltau]
        B = e^V4/2 ... e^V1/2 e^V1/2 ... e^V4/2
        return |I + BLtau B_Ltau-1 ... B1|
        """
        assert boson.size(0) == 1
        boson = boson.squeeze(0)

        Vs = self.Lx * self.Ly
        dtau = self.dtau
        t = self.t
        boson = boson.permute([3, 2, 1, 0]).reshape(-1)

        prodB = torch.eye(Vs, device=boson.device, dtype=torch.complex64)
        for tau in reversed(range(self.Ltau)):
            dB1 = self.get_dB1()
            dB1[self.i_list_1, self.j_list_1] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1[self.j_list_1, self.i_list_1] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            B = dB1

            dB = self.get_dB()
            dB[self.i_list_2, self.j_list_2] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB[self.j_list_2, self.i_list_2] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB

            dB = self.get_dB()
            dB[self.i_list_3, self.j_list_3] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB[self.j_list_3, self.i_list_3] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB

            dB = self.get_dB()
            dB[self.i_list_4, self.j_list_4] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB[self.j_list_4, self.i_list_4] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB

            prodB = prodB @ B

        I = torch.eye(Vs, device=boson.device, dtype=torch.complex64)
        detM = torch.det(I + prodB)
        return detM


    def action_boson(self, boson):
        """
        The E_b = 1/2 * boson.transpose * self.A * boson.

        :param momentum: [bs, 2, Lx, Ly, Ltau] tensor
        :param boson: [bs, 2, Lx, Ly, Ltau] tensor
        :return: the action
        """
        force = -torch.einsum('ijklmnop,bmnop->bijkl', self.A, boson)
        potential = 0.5 * torch.einsum('bijkl,bijkl->b', boson, force)
        return potential   

    def action_boson_plaq(self, boson):
        """
        boson: [bs, 2, Lx, Ly, Ltau]
        S: [bs,]
        """
        boson = boson.permute([3, 2, 1, 4, 0]).reshape(-1, self.Ltau, self.bs)
        curl = torch.einsum('ij,jkl->ikl', self.curl_mat, boson)  # [Vs, Ltau, bs]
        S = self.K * torch.sum(torch.cos(curl), dim=(0, 1))  
        return S   
    
    # def action_boson_plaq_noncomp(self, boson):
    #     """
    #     boson: [bs, 2, Lx, Ly, Ltau]
    #     S: [bs,]
    #     """         
    #     boson = boson.permute([3, 2, 1, 4, 0]).reshape(-1, self.Ltau, self.bs)
    #     curl = torch.einsum('ij,jkl->ikl', self.curl_mat, boson)  # [Vs, Ltau, bs]
    #     S = self.K/2 * torch.sum(curl**2, dim=(0, 1))
    #     return S
    
    # @torch.inference_mode()
    @torch.no_grad()
    def measure(self):
        """
        boson: [2, Lx, Ly, Ltau]

        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        # Initialization
        self.initialize_boson_staggered_pi()
        # self.initialize_boson()
        self.G_list[-1] = self.sin_curl_greens_function_batch(self.boson)
        self.S_tau_list[-1] = self.action_boson_tau_cmp(self.boson)
    
        self.boson_energy = self.action_boson_tau_cmp(self.boson) + self.action_boson_plaq(self.boson)
        detM = self.get_detM(self.boson)
        # detM_np = self.get_detM_np(self.boson)
        self.fermion_energy = -2 * torch.log(torch.real(detM)).unsqueeze(0)

        # Measure
        plt.figure()
        # Take sample
        data_folder = script_path + "/check_points/local_check_point/"
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp, energies_old, energies_new = self.local_u1_proposer()
            Sf0, Sb0, H0 = energies_old
            Sf_fin, Sb_fin, H_fin = energies_new
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
            self.G_list[i] = \
                accp.view(-1, 1) * self.sin_curl_greens_function_batch(boson) \
              + (1 - accp.view(-1, 1).to(torch.float)) * self.G_list[i-1]
            self.S_tau_list[i] = \
                accp.view(-1, 1) * self.action_boson_tau_cmp(boson) \
              + (1 - accp.view(-1, 1).to(torch.float)) * self.S_tau_list[i-1]
            self.S_plaq_list[i] = torch.where(accp, Sb_fin, Sb0)
            self.Sf_list[i] = torch.where(accp, Sf_fin, Sf0)

            self.step += 1
            self.cur_step += 1
            
            # plotting
            if i % (self.Vs * 10) == 0:
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)  # Pause for 5 seconds
                plt.close()

            # checkpointing
            if i % (self.Vs * 10) == 0:
                res = {'boson': boson,
                    'step': self.step,
                    'G_list': self.G_list.cpu(),
                    'S_plaq_list': self.S_plaq_list.cpu(),
                    'S_tau_list': self.S_tau_list.cpu(),
                    'Sf_list': self.Sf_list.cpu()}     
                
                data_folder = script_path + "/check_points/local_check_point/"
                file_name = f"ckpt_N_{self.specifics}_step_{self.step}"
                self.save_to_file(res, data_folder, file_name)           

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'G_list': self.G_list.cpu(),
               'S_plaq_list': self.S_plaq_list.cpu(),
               'S_tau_list': self.S_tau_list.cpu(),
               'Sf_list': self.Sf_list.cpu()}        

        # Save to file
        file_name = f"ckpt_N_{self.specifics}_step_{self.step}"
        self.save_to_file(res, data_folder, file_name)           

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
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        start = 50  # to prevent from being out of scale due to init out-liers
 
        axes[2].plot(self.accp_rate[start:self.cur_step].cpu().numpy())
        axes[2].set_xlabel("Steps")
        axes[2].set_ylabel("Acceptance Rate")

        idx = [0, self.num_tau // 2, -2]
        # for b in range(self.G_list.size(1)):
        axes[1].plot(self.G_list[start:self.cur_step, ..., idx[0]].mean(axis=1).cpu().numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[start:self.cur_step, ..., idx[1]].mean(axis=1).cpu().numpy(), label=f'G[{self.num_tau // 2}]')
        axes[1].plot(self.G_list[start:self.cur_step, ..., idx[2]].mean(axis=1).cpu().numpy(), label=f'G[-2]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

        # axes[2].plot(self.H_list[self.N_therm_step:].cpu().numpy())
        # axes[2].set_ylabel("H")
        axes[0].plot(self.S_plaq_list[self.N_therm_step + start: self.N_therm_step + self.cur_step].cpu().numpy(), 'o')
        axes[0].plot(self.Sf_list[self.N_therm_step + start: self.N_therm_step + self.cur_step].cpu().numpy(), '*')
        axes[0].set_ylabel("S")


        plt.tight_layout()
        # plt.show(block=False)

        class_name = __file__.split('/')[-1].replace('.py', '')
        method_name = "totol_monit"
        save_dir = os.path.join(script_path, f"./figures/local_{class_name}")
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


    filename = script_path + f"/check_points/local_check_point/ckpt_N_{specifics}_step_{step}.pt"
    res = torch.load(filename)
    print(f'Loaded: {filename}')

    G_list = res['G_list']
    step = res['step']
    x = np.array(list(range(G_list[0].size(-1))))

    start = 500
    end = step
    sample_step = 1
    seq_idx = np.arange(start, end, sample_step)

    ## Plot
    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_avg', color='blue', lw=2)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={end - start}")

    # Save plot
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens"
    save_dir = os.path.join(script_path, f"./figures/local_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


    # -------- Log plot -------- #
    plt.figure()
    plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))

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


    # --------- save_plot ---------
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens_loglog"
    save_dir = os.path.join(script_path, f"./figures/local_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


if __name__ == '__main__':

    J = float(os.getenv("J", '0.5'))
    Nstep = int(os.getenv("Nstep", '10'))
    print(f'J={J} \nNstep={Nstep}')

    hmc = LocalUpdateSampler(J=J, Nstep=Nstep)
    # hmc = LocalUpdateSampler()

    # Measure
    G_avg, G_std = hmc.measure()

    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc.N_step, hmc.specifics, False)

    plt.show()

    exit()

