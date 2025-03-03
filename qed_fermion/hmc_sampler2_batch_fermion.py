import json
import math
import matplotlib
import numpy as np
matplotlib.use('MacOSX')
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')

import os

from qed_fermion.coupling_mat3 import initialize_coupling_mat3, initialize_curl_mat

script_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"device: {device}")
dtype = torch.float32

class HmcSampler(object):
    def __init__(self, config=None):
        # Dims
        self.Lx = 8
        self.Ly = 8
        self.Ltau = 8
        self.bs = 1

        # Couplings
        self.dtau = 0.25
        self.J = 1
        self.K = 1
        self.t = 1

        self.boson = None
        # self.A = initialize_coupling_mat3(self.Lx, self.Ly, self.Ltau, self.J, self.dtau, self.K)[0]
        # assert self.A[0, 0, 0, 0, 0, 0, 0, 0].item() > 0, f'incorrect input of A'
        # self.A = self.A.to(device)
        # assert self.A[0, 0, 0, 0, 0, 0, 0, 0].item() > 0, f'device: {device} may have silently failed on large tensor'
        self.curl_mat = initialize_curl_mat(self.Lx, self.Ly).to(device)

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 0
        self.N_step = int(1e3) + 100
        self.step = 0
        self.cur_step = 0
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.bs, self.num_tau + 1, device=device)
        self.accp_list = torch.zeros(self.N_step, self.bs, dtype=torch.bool, device=device)
        self.accp_rate = torch.zeros(self.N_step, self.bs, device=device)
        self.S_list = torch.zeros(self.N_therm_step + self.N_step, self.bs)
        self.H_list = torch.zeros(self.N_therm_step + self.N_step, self.bs)

        # Leapfrog
        self.m = 1/2 * 4
        # self.delta_t = 0.05
        self.delta_t = 0.5
        # m = 2, T(~sqrt(m/k)) = 1.5, N = T / 0.05 = 30
        # m = 1/2, T = 0.75, N = T / 0.05 = 15
        # T/4 = 0.375

        # total_t = 0.375
        # self.N_leapfrog = int(total_t // self.delta_t)
        # 1st exp: t = 1, delta_t = 0.05, N_leapfrog = 20

        self.N_leapfrog = 5
        # Fixing the total number of leapfrog step, then the larger delta_t, the longer time the Hamiltonian dynamic will reach, the less correlated is the proposed config to the initial config, where the correlation is in the sense that, in the small delta_t limit, almost all accpeted and p being stochastic, then the larger the total_t, the less autocorrelation. But larger delta_t increases the error amp and decreases the acceptance rate.

        # Increasing m, say by 4, the sigma(p) increases by 2. omega = sqrt(k/m) slows down by 2 [cos(wt) ~ 1 - 1/2 * k/m * t^2]. The S amplitude is not affected (since it's decided by initial cond.), but somehow H amplitude decreases by 4, similar to omega^2 decreases by 4. 

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False

        # Initialization
        self.initialize_geometry()

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
        self.M = torch.eye(Vs*self.Ltau, device=device, dtype=torch.complex64)

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
       
    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [bs, 2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * math.sqrt(self.m)

    def draw_psudo_fermion(self):
        """
        Draw psudo_fermion psi = M(x0)'R 
        :return: [bs, Lx * Ly * Ltau] gaussian tensor
        """
        R = torch.randn(self.bs, self.Lx * self.Ly * self.Ltau, device=device) / math.sqrt(2) + 1j * torch.randn(self.bs, self.Lx * self.Ly * self.Ltau, device=device) / math.sqrt(2)
        return R
    
    def force(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [bs, 2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x.
        """
        return -torch.einsum('ijklmnop,bmnop->bijkl', self.A, x)

    def metropolis_update(self):
        """
        Perform one step of metropolis update. Update self.boson.

        Given the last boson (conditional on the past) and momentum (iid sampled), the join dist. is the desired one. Then, the leapfrog proposes new config and the metropolis update preserves the join dist. The marginal dist. of the config is always conditional on the past while the momentum is not. Kinetic + potential (action) is conserved in the Hamiltonian dynamics but the action is not.

        :return: None
        """
        boson_new, energies_old, energies_new = self.leapfrog_proposer3()
        Sf_new, Sb_new, H_new = energies_new
        Sf_old, Sb_old, H_old = energies_old
        print(f"H_old, H_new, diff: {H_old}, {H_new}, {H_new - H_old}")
        print(f"threshold: {torch.exp(H_old - H_new).item()}")

        accp = torch.rand(self.bs, device=device) < torch.exp(H_old - H_new)
        print(f'Accp?: {accp.item()}')
        self.boson[accp] = boson_new[accp]
        return self.boson, accp, energies_old, energies_new

    def curl_greens_function_batch(self, boson):
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

        curl_phi = (
            phi_x
            + torch.roll(phi_y, shifts=-1, dims=1)  # y-component at (x+1, y)
            - torch.roll(phi_x, shifts=-1, dims=2)  # x-component at (x, y+1)
            - phi_y
        )  # Shape: [batch_size, Lx, Ly, Ltau]

        correlations = []
        for dtau in range(self.num_tau + 1):
            idx1 = list(range(Ltau))
            idx2 = [(i + dtau) % Ltau for i in idx1]
            
            corr = torch.mean(curl_phi[..., idx1] * curl_phi[..., idx2], dim=(1, 2, 3))
            correlations.append(corr)

        return torch.stack(correlations).T  # Shape: [bs, num_dtau]

    # =========== Turn on fermions =========
    def harmonic_evol(self, p, x, delta_t):
        A = self.A

        return p, x
    
    def action_boson_tau(self, x):
        """
        x:  [bs, 2, Lx, Ly, Ltau]

        S = 1/2 * x' @ A @ x
            F_{r,k} = e^{irk}
            F'F/L = FF'/L = i.d.
            x_k = fft(x) = F'@x = sum_r e^{-irk} x_r
            x_r = ifft(xk) = 1/L * F @ x_k = 1/L * sum_k e^{ikr} x_k

        S = 1/2 * x'@F @ F'@A@F @ F'@x / L**2
          = 1/2 * xk' @ F'F diag(A_k) @ xk / L**2
          = 1/2 * xk' @ diag(A_k) @ xk / L

        k = torch.fft.fftfreq(L) = [\kap/L] in the unit of 2*torch.pi
        A_k = 2 * (1 - torch.cos(k * 2*torch.pi))

        """
        coeff = 1/2 / self.J / self.dtau**2
        xk = torch.fft.fft(x, dim=-1)
        L = self.Ltau
        k = torch.fft.fftfreq(L).to(device)    
        lmd = 2 * (1 - torch.cos(k * 2*torch.pi))[None, None, None, None]

        # F = torch.fft.fft(torch.eye(3), dim=0)
        # torch.eye(3) = 1/L * F.conj().T @ F
        return coeff * (xk.abs()**2 * lmd).sum(dim=(1, 2, 3, 4)) / L

    
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

        omega = torch.sqrt(1/self.m * 2 * (1 - torch.cos(k * 2*torch.pi)) / self.J / self.dtau**2)

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

    def df_dot(self, boson_list, xi_lft, xi_rgt, i_list, j_list, is_group_1=False):
        """
        dB_{is, js}: (i, j, v), (j, i, v')
        """
        dtau = self.dtau
        t = self.t
        v = (math.sinh(t * dtau/2) if not is_group_1 else math.sinh(t * dtau)) * \
            torch.exp(1j * (boson_list + torch.pi/2))
        res = xi_lft[i_list] * xi_rgt[j_list] * v
        res += xi_lft[j_list] * xi_rgt[i_list] * v.conj()
        return res

    def force_f(self, psi, Mt, boson, xi_init=None):
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
        psi = psi.T

        Ot = Mt.T.conj() @ Mt
        L = torch.linalg.cholesky(Ot)
        Ot_inv = torch.cholesky_inverse(L)
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
            for mat_B in reversed([self.B4, self.B3, self.B2, self.B1, self.B2, self.B3, self.B4]):
                B_xi = mat_B @ B_xi   
            B_xi = B_xi.T.conj()       

            xi_n_lft_5 = xi_n.T.conj().view(-1)
            xi_n_lft_4 = (xi_n_lft_5 @ self.B4).view(-1)
            xi_n_lft_3 = (xi_n_lft_4 @ self.B3).view(-1)
            xi_n_lft_2 = (xi_n_lft_3 @ self.B2).view(-1)
            xi_n_lft_1 = (xi_n_lft_2 @ self.B1).view(-1)
            xi_n_lft_0 = (xi_n_lft_1 @ self.B2).view(-1)           
            xi_n_lft_m1 = (xi_n_lft_0 @ self.B3).view(-1)

            xi_c_rgt_5 = xi_c.view(-1)
            xi_c_rgt_4 = (self.B4 @ xi_c_rgt_5).view(-1)
            xi_c_rgt_3 = (self.B3 @ xi_c_rgt_4).view(-1) 
            xi_c_rgt_2 = (self.B2 @ xi_c_rgt_3).view(-1) 
            xi_c_rgt_1 = (self.B1 @ xi_c_rgt_2).view(-1) 
            xi_c_rgt_0 = (self.B2 @ xi_c_rgt_1).view(-1)          
            xi_c_rgt_m1 = (self.B3 @ xi_c_rgt_0).view(-1) 

            B_xi_5 = B_xi.view(-1)
            B_xi_4 = (B_xi_5 @ self.B4).view(-1)
            B_xi_3 = (B_xi_4 @ self.B3).view(-1)
            B_xi_2 = (B_xi_3 @ self.B2).view(-1)
            B_xi_1 = (B_xi_2 @ self.B1).view(-1)
            B_xi_0 = (B_xi_1 @ self.B2).view(-1)
            B_xi_m1 = (B_xi_0 @ self.B3).view(-1)

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
            + self.df_dot(boson_list, xi_n_lft_1, xi_c_rgt_3, self.i_list_2, self.j_list_2)
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
        return Ft, xi_t.view(self.bs, -1)

        # # precondition
        # Ot_inv = self.O_inv

        # # CG_Solve
        # # xi_t = torch.cholesky_solve(psi, Mt.T.conj())
        # xi_t = self.cg_solver(Ot, psi, Ot_inv, xi_init)

        # # Compute force
        # Ff_t = - xi_t.T.conj() @ force_mat @ xi_t

        return Ff_t, xi_t

    def cg_solver(self, Ot, psi, Ot_inv, xi_init=None):
        if xi_init is None:
            xi_init = Ot_inv @ psi

        xi_t = None
        return xi_t
    

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

        for tau in range(self.Ltau):
            dB1 = self.get_dB1()
            dB1[self.i_list_1, self.j_list_1] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            dB1[self.j_list_1, self.i_list_1] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_1]) * math.sinh(t * dtau)
            B = dB1
            self.B1 = dB1

            dB = self.get_dB()
            dB[self.i_list_2, self.j_list_2] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            dB[self.j_list_2, self.i_list_2] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_2]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB
            self.B2 = dB

            dB = self.get_dB()
            dB[self.i_list_3, self.j_list_3] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            dB[self.j_list_3, self.i_list_3] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_3]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB
            self.B3 = dB

            dB = self.get_dB()
            dB[self.i_list_4, self.j_list_4] = \
                torch.exp(1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            dB[self.j_list_4, self.i_list_4] = \
                torch.exp(-1j * boson[2*Vs*tau + self.boson_idx_list_4]) * math.sinh(t * dtau/2)
            B = dB @ B @ dB
            self.B4 = dB

            if tau < self.Ltau - 1:
                row_start = Vs * (tau + 1)
                row_end = Vs * (tau + 2)
                col_start = Vs * tau
                col_end = Vs * (tau + 1)
                self.M[row_start:row_end, col_start:col_end] = -B
            else:
                self.M[:Vs, Vs*tau:] = B

        return self.M


    def leapfrog_proposer3(self):
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
        x = self.boson  # [bs, 2, Lx, Ly, Ltau] tensor
        p = p0

        R = self.draw_psudo_fermion()

        # Initial energies
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            M_auto = self.get_M(x)
            psi = torch.einsum('rs,bs->br', M_auto.T.conj(), R)

            Ot = M_auto.conj().T @ M_auto
            L = torch.linalg.cholesky(Ot)
            O_inv = torch.cholesky_inverse(L) 

            Sf = torch.einsum('bi,ij,bj->b', psi.conj(), O_inv, psi)
            torch.testing.assert_close(torch.imag(Sf), torch.zeros_like(torch.imag(Sf)))
            Sf0 = torch.real(Sf)
            force_f = -torch.autograd.grad(Sf0, x, create_graph=False)[0]
        
        assert x.grad is None

        Sb0 = self.action_boson_tau(x) + self.action_boson_plaq(x)
        H0 = Sf0 + Sb0 + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)

        print(f"Sb_tau={self.action_boson_tau(x)}")
        print(f"p**2={torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)}")
        if self.debug_pde:
            b_idx = 0
            
            # Initialize plot
            # plt.ion()  # Turn on interactive mode
            # fig, ax = plt.subplots()
            fig, axs = plt.subplots(2, 1, figsize=(6, 8))  # Two rows, one column

            Hs = [H0[b_idx].item()]
            Ss = [(Sf0 + Sb0)[b_idx].item()] 
            # Ss = [Sb0[b_idx].item()] 

            # Setup for first subplot (Hs)
            line_Hs, = axs[0].plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
            axs[0].set_ylabel('Hamiltonian (H)')
            axs[0].legend()
            axs[0].grid()

            # Setup for second subplot (Ss)
            # axs[1].set_title('Real-Time Evolution of S_s')
            line_Ss, = axs[1].plot(Ss, marker='s', linestyle='-', color='r', label='S_s')
            axs[1].set_xlabel('Leapfrog Step')
            axs[1].set_ylabel('S_b')
            axs[1].legend()
            axs[1].grid()
    
        # Multi-scale Leapfrog
        # H(x, p) = U1/2 + sum_m (U0/2M + K/M + U0/2M) + U1/2 
         
        dt = self.delta_t
        # force_f, xi_t = self.force_f(psi, M0, x)

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Sb_plaq = self.action_boson_plaq(x)
            force_b = -torch.autograd.grad(Sb_plaq, x, create_graph=False)[0]

        for leap in range(self.N_leapfrog):

            p = p + dt/2 * force_f

            # Update (p, x)
            M = 5
            for _ in range(M):
                p = p + force_b * dt/2/M
                x, p = self.harmonic_tau(x, p, dt/M)

                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    Sb_plaq = self.action_boson_plaq(x)
                    force_b = -torch.autograd.grad(Sb_plaq, x, create_graph=False)[0]
            
                p = p + force_b * dt/2/M

            force_f, xi_t = self.force_f(psi, self.get_M(x), x)
            with torch.enable_grad():
                x = x.clone().requires_grad_(True)
                M_auto = self.get_M(x)
                psi = torch.einsum('rs,bs->br', M_auto.T.conj(), R)

                Ot = M_auto.conj().T @ M_auto
                L = torch.linalg.cholesky(Ot)
                O_inv = torch.cholesky_inverse(L) 

                Sf = torch.einsum('bi,ij,bj->b', psi.conj(), O_inv, psi)
                torch.testing.assert_close(torch.imag(Sf), torch.zeros_like(torch.imag(Sf)), atol=1e-3, rtol=1e-3)
                Sf = torch.real(Sf)
                force_f = -torch.autograd.grad(Sf, x, create_graph=False)[0]

            p = p + dt/2 * force_f

            if self.debug_pde:
                Sb_t = self.action_boson_plaq(x) + self.action_boson_tau(x)
                H_t = Sf + Sb_t + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)

                torch.testing.assert_close(H0, H_t, atol=0.1, rtol=0.05)

                print(leap)
                print(f"Sb_tau={self.action_boson_tau(x)}")
                print(f"p**2={torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)}")
                print(f'H={H_t}')

                # Hd, Sd = self.action((p + p_last)/2, x)  # Append new H value   
                Hs.append(H_t[b_idx].item())
                Ss.append((Sf + Sb_t)[b_idx].item())

                # Update data for both subplots
                line_Hs.set_data(range(len(Hs)), Hs)
                line_Ss.set_data(range(len(Ss)), Ss)

                # Adjust limits dynamically
                axs[0].relim()
                axs[0].autoscale_view()
                amp = max(Hs) - min(Hs)
                axs[0].set_title(f'dt={self.delta_t:.2f}, m={self.m}, amp={amp:.2g}, amp_rtol={amp/sum(Hs)*len(Hs):.2g}, N={self.N_leapfrog}')

                axs[1].relim()
                axs[1].autoscale_view()
                amp = max(Ss) - min(Ss)
                axs[1].set_title(f'dt={self.delta_t:.3f}, m={self.m}, amp={amp:.2g}, N={self.N_leapfrog}') 

                plt.pause(0.1)   # Small delay to update the plot
                
                # --------- save_plot ---------
                if leap == self.N_leapfrog - 1:
                    class_name = __file__.split('/')[-1].replace('.py', '')
                    method_name = "fermion_couple"
                    save_dir = os.path.join(script_path, f"./figures/figure_leapfrog")
                    os.makedirs(save_dir, exist_ok=True) 
                    file_path = os.path.join(save_dir, f"{method_name}_Lx_{self.Lx}_Ltau_{self.Ltau}_Jtau_{self.J * self.dtau**2}_K_{self.K}_t_{self.t}_Nleap_{self.N_leapfrog}_dt_{self.delta_t}.pdf")
                    plt.savefig(file_path, format="pdf", bbox_inches="tight")
                    print(f"Figure saved at: {file_path}")


        # Final energies
        # Sf_fin = torch.einsum('br,br->b', psi.conj(), xi_t)
        # torch.testing.assert_close(torch.imag(Sf_fin).view(-1).cpu(), torch.tensor([0], dtype=torch.float32))
        # Sf_fin = torch.real(Sf_fin)
        Sb_fin = self.action_boson_plaq(x) + self.action_boson_tau(x) 
        H_fin = Sf + Sb_fin + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
 
        # torch.testing.assert_close(H0, H_fin, atol=1e-3, rtol=0.05)

        return x, (Sf0, Sb0, H0), (Sf, Sb_fin, H_fin)


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
    
    # @torch.inference_mode()
    @torch.no_grad()
    def measure(self):
        """
        boson: [2, Lx, Ly, Ltau]

        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        self.initialize_boson_staggered_pi()
        self.G_list[-1] = self.curl_greens_function_batch(self.boson)

        # # Thermalize
        # self.thrm_bool = True
        # boson = None
        # for i in range(self.N_therm_step):
        #     boson, accp, H, S = self.metropolis_update()
        #     self.H_list[i] = H
        #     self.S_list[i] = S
        # self.G_list[-1] = self.curl_greens_function_batch(boson)

        plt.figure()
        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp, energies_old, energies_new = self.metropolis_update()
            Sf0, Sb0, H0 = energies_old
            Sf_fin, Sb_fin, H_fin = energies_new
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
            self.G_list[i] = accp.view(-1, 1) * self.curl_greens_function_batch(boson) + (1 - accp.view(-1, 1).to(torch.float)) * self.G_list[i-1]
            self.S_list[i] = torch.where(accp, Sf_fin, Sf0)

            self.step += 1
            self.cur_step += 1
            
            # plotting
            if i % 200 == 0:
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)  # Pause for 5 seconds
                plt.close()

            # checkpointing
            if i % 200 == 0:
                res = {'boson': boson,
                       'step': self.step,
                       'mass': self.m,
                       'G_list': self.G_list.cpu()}
                
                data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point/"
                file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
                self.save_to_file(res, data_folder, file_name)           

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'mass': self.m,
               'G_avg': G_avg,
               'G_std': G_std,
               'G_list': self.G_list.cpu()}

        # Save to file
        data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point/"
        file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
        self.save_to_file(res, data_folder, file_name)           

        return G_avg, G_std

    # ------- Save to file -------
    def save_to_file(self, res, data_folder, filename):
        os.makedirs(data_folder, exist_ok=True)
        filepath = os.path.join(data_folder, f"{filename}.pt")
        # with open(filepath, "wb") as f:
        #     pickle.dump(res, f)
        torch.save(res, filepath)
        print(f"Data saved to {data_folder + filepath}")

    # ------- Visualization -------
    def total_monitoring(self):
        """
        Visualize obsrv and accp in subplots.
        """
        # plt.figure()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        start = 50

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
        axes[0].plot(self.S_list[self.N_therm_step + start: self.N_therm_step + self.cur_step].cpu().numpy(), 'o')
        axes[0].set_ylabel("S")


        plt.tight_layout()
        # plt.show(block=False)

        class_name = __file__.split('/')[-1].replace('.py', '')
        method_name = "totol_monit"
        save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}_{self.Ly}_Ltau_{self.Ltau}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")


def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), step=1000001):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/hmc_check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    res = torch.load(filename)

    G_list = res['G_list']
    step = res['step']
    x = np.array(list(range(G_list[0].size(-1))))

    start = 300
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
    save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


    # -------- Log plot -------- #
    plt.figure()
    plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))

    # Analytical data
    data_folder_anal = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
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
    save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


if __name__ == '__main__':

    hmc = HmcSampler()
    G_avg, G_std = hmc.measure()

    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    # Lx, Ly, Ltau = 30, 30, 20
    step = 1100
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), step)

    plt.show()
    dbstop = 1
