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

class HmcSampler(object):
    def __init__(self, J=0.5, Nstep=3000, config=None):
        # Dims
        self.Lx = 6
        self.Ly = 6
        self.Ltau = 10
        self.bs = 1

        # Couplings
        self.Nf = 2
        # J = 0.5  # 0.5, 1, 3
        # self.dtau = 2/(J*self.Nf)

        self.dtau = 0.1
        scale = self.dtau  # used to apply dtau
        self.J = J / scale * self.Nf / 4
        self.K = 1 * scale * self.Nf
        self.t = 0.001

        # t * dtau < const
        # fix t, increase J

        # J,t = (4, 0.1) 40:  Ff=4, Fb=9.7, delta_t=0.2
        # J,t = (20, 0.5) 40: Ff=4, Fb=9.7, delta_t=0.2
        # J,t = (40, 1) 40:   Ff=4, Fb=1, delta_t=2

        self.boson = None
        self.boson_energy = None

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 0
        self.N_step = int(Nstep)
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

        # Leapfrog
        self.m = 1/2 * 4 / scale
        # self.m = 1/2
        # self.delta_t = min(0.05 / scale, 0.01)
        self.delta_t = 0.02
        print(f"delta_t = {self.delta_t}")

        # self.N_leapfrog = 12 if J >0.99 else 6
        # self.N_leapfrog = 6
        # self.N_leapfrog = 15 * 40
        self.N_leapfrog = 15

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
        self.specifics = f"cmp_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_dtau_{self.dtau}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf:.2g}_t_{self.t}_Nleap_{self.N_leapfrog}_dt_{self.delta_t}"
    
    def get_specifics(self):
        return f"cmp_{self.Lx}_Ltau_{self.Ltau}_Nstp_{self.N_step}_dtau_{self.dtau}_Jtau_{self.J*self.dtau/self.Nf*4:.2g}_K_{self.K/self.dtau/self.Nf:.2g}_t_{self.t}_Nleap_{self.N_leapfrog}_dt_{self.delta_t}"

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
    

    def metropolis_update(self):
        """
        Perform one step of metropolis update. Update self.boson.

        Given the last boson (conditional on the past) and momentum (iid sampled), the join dist. is the desired one. Then, the leapfrog proposes new config and the metropolis update preserves the join dist. The marginal dist. of the config is always conditional on the past while the momentum is not. Kinetic + potential (action) is conserved in the Hamiltonian dynamics but the action is not.

        :return: None
        """
        boson_new, energies_old, energies_new = self.leapfrog_proposer4_cmptau()
        H_new = energies_new[-1]
        H_old = energies_old[-1]
        # print(f"H_old, H_new, diff: {H_old}, {H_new}, {H_new - H_old}")
        # print(f"threshold: {torch.exp(H_old - H_new).item()}")

        accp = torch.rand(self.bs, device=device) < torch.exp(H_old - H_new)
        # print(f'Accp?: {accp.item()}')
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
        x = self.boson  # [bs, 2, Lx, Ly, Ltau] tensor
        p = p0

        Sb0 = self.action_boson_tau_cmp(x) + self.action_boson_plaq(x)
        H0 = Sb0 + torch.sum(p0 ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)

        dt = self.delta_t

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            Sb_plaq = self.action_boson_plaq(x)
            force_b = -torch.autograd.grad(Sb_plaq, x, create_graph=False)[0]

            assert x.grad is None
            Sb_tau = self.action_boson_tau_cmp(x)
            force_b_tau = -torch.autograd.grad(Sb_tau, x, create_graph=False)[0]

        if self.debug_pde:
            # print(f"Sb_tau={self.action_boson_tau(x)}")
            # print(f"p**2={torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)}")

            b_idx = 0
            
            # Initialize plot
            # plt.ion()  # Turn on interactive mode
            # fig, ax = plt.subplots()
            fig, axs = plt.subplots(3, 1, figsize=(6, 8))  # Two rows, one column

            Hs = [H0[b_idx].item()]
            Ss = [(Sb0)[b_idx].item()]
            force_bs = [torch.linalg.norm(force_b.reshape(self.bs, -1), dim=1)[b_idx].item()]
            # Ss = [Sb0[b_idx].item()]

            # Setup for 1st subplot (Hs)
            line_Hs, = axs[0].plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
            axs[0].set_ylabel('Hamiltonian (H)')
            axs[0].set_xticks([])  # Remove x-axis ticks
            axs[0].legend()
            axs[0].grid()

            # Setup for 2nd subplot (Ss)
            # axs[1].set_title('Real-Time Evolution of S_s')
            line_Ss, = axs[1].plot(Ss, marker='s', linestyle='-', color='r', label='S_s')
            # axs[1].set_xlabel('Leapfrog Step')
            axs[1].set_ylabel('S_b')
            axs[1].set_xticks([])  # Remove x-axis ticks
            axs[1].legend()
            axs[1].grid()

            # Setup for 3rd subplot (force)
            # axs[1].set_title('Real-Time Evolution of S_s')
            line_force_b, = axs[2].plot(force_bs, marker='s', linestyle='-', color='b', label='force_b')
            axs[2].set_xlabel('Leapfrog Step')
            axs[2].set_ylabel('forces_norm')
            axs[2].legend()
            axs[2].grid()

        # Multi-scale Leapfrog
        # H(x, p) = U1/2 + sum_m (U0/2M + K/M + U0/2M) + U1/2 

        for leap in range(self.N_leapfrog):

            # p = p + dt/2 * (force_f_u + force_f_d)

            # Update (p, x)
            M = 5
            for _ in range(M):
                # p = p + force(x) * dt/2
                # x = x + velocity(p) * dt
                # p = p + force(x) * dt/2

                p = p + (force_b + force_b_tau) * dt/2/M
                x = x + p / self.m * dt/M
                
                with torch.enable_grad():
                    x = x.clone().requires_grad_(True)
                    Sb_plaq = self.action_boson_plaq(x)
                    force_b = -torch.autograd.grad(Sb_plaq, x, create_graph=False)[0]
                    
                    assert x.grad is None
                    Sb_tau = self.action_boson_tau_cmp(x)
                    force_b_tau = -torch.autograd.grad(Sb_tau, x, create_graph=False)[0]

                p = p + (force_b + force_b_tau) * dt/2/M

            # Mt, B_list = self.get_M(x)
            # (force_f_u, force_f_d), (xi_t_u, xi_t_d) = self.force_f([psi_u, psi_d], Mt, x, B_list)
            # p = p + dt/2 * (force_f_u + force_f_d)

            if self.debug_pde:
                Sb_t = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x)
                H_t = Sb_t + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)

                torch.testing.assert_close(H0, H_t, atol=1e-1, rtol=5e-2)

                # print(leap)
                # print(f"Sb_tau={self.action_boson_tau(x)}")
                # print(f"p**2={torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)}")
                # print(f'H={H_t}')

                # Hd, Sd = self.action((p + p_last)/2, x)  # Append new H value   
                Hs.append(H_t[b_idx].item())
                Ss.append((Sb_t)[b_idx].item())
                force_bs.append(torch.linalg.norm(force_b.reshape(self.bs, -1), dim=1)[b_idx].item())

                # Update data for both subplots
                line_Hs.set_data(range(len(Hs)), Hs)
                line_Ss.set_data(range(len(Ss)), Ss)
                line_force_b.set_data(range(len(force_bs)), force_bs)

                # Adjust limits dynamically
                axs[0].relim()
                axs[0].autoscale_view()
                amp = max(Hs) - min(Hs)
                axs[0].set_title(f'dt={self.delta_t:.2f}, m={self.m}, atol={amp:.2g}, rtol={amp/sum(Hs)*len(Hs):.2g}, N={self.N_leapfrog}')

                axs[1].relim()
                axs[1].autoscale_view()
                amp = max(Ss) - min(Ss)
                axs[1].set_title(f'dt={self.delta_t:.3f}, m={self.m}, atol={amp:.2g}, N={self.N_leapfrog}') 

                axs[2].relim()
                axs[2].autoscale_view()
                axs[2].set_title(f'mean_force_b={sum(force_bs)/len(force_bs):.2g}, mean_force_f={sum(force_bs)/len(force_bs):.2g}')

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
        Sb_fin = self.action_boson_plaq(x) + self.action_boson_tau_cmp(x) 
        H_fin = Sb_fin + torch.sum(p ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
 
        # torch.testing.assert_close(H0, H_fin, atol=5e-3, rtol=0.05)

        return x, (0, Sb0, H0), (0, Sb_fin, H_fin)


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
        # Initialization
        self.initialize_boson_staggered_pi()
        # self.initialize_boson()
        self.G_list[-1] = self.sin_curl_greens_function_batch(self.boson)
        self.S_plaq_list[-1] = self.action_boson_plaq(self.boson)
        self.S_tau_list[-1] = self.action_boson_tau_cmp(self.boson)

        # Sb = self.action_boson_tau_cmp(self.boson) + self.action_boson_plaq(self.boson)
        # detM = self.get_detM(self.boson)
        # self.boson_energy = Sb + 2 * torch.log(torch.real(detM))

        # Measure
        plt.figure()
        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp, energies_old, energies_new = self.metropolis_update()
            Sf0, Sb0, H0 = energies_old
            Sf_fin, Sb_fin, H_fin = energies_new
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
            self.G_list[i] = \
                accp.view(-1, 1) * self.sin_curl_greens_function_batch(boson) \
              + (1 - accp.view(-1, 1).to(torch.float)) * self.G_list[i-1]
            self.S_plaq_list[i] = \
                accp.view(-1, 1) * self.action_boson_plaq(boson) \
              + (1 - accp.view(-1, 1).to(torch.float)) * self.S_plaq_list[i-1]
            self.S_tau_list[i] = \
                accp.view(-1, 1) * self.action_boson_tau_cmp(boson) \
              + (1 - accp.view(-1, 1).to(torch.float)) * self.S_tau_list[i-1]
            # self.Sf_list[i] = torch.where(accp, Sf_fin, Sf0)

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
                    'G_list': self.G_list.cpu(),
                    'S_plaq_list': self.S_plaq_list.cpu(),
                    'S_tau_list': self.S_tau_list.cpu(),
                    'Sf_list': self.Sf_list.cpu()}
                
                data_folder = script_path + "/check_points/hmc_check_point/"
                file_name = f"ckpt_N_{self.specifics}_step_{self.step}"
                self.save_to_file(res, data_folder, file_name)           

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'mass': self.m,
               'G_list': self.G_list.cpu(),
               'S_plaq_list': self.S_plaq_list.cpu(),
               'S_tau_list': self.S_tau_list.cpu(),
               'Sf_list': self.Sf_list.cpu()}

        # Save to file
        data_folder = script_path + "/check_points/hmc_check_point/"
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
        save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
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

    if len(specifics) == 0:
        filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    else:
        filename = script_path + f"/check_points/hmc_check_point/ckpt_N_{specifics}_step_{step}.pt"
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
    save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
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
    save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{specifics}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


if __name__ == '__main__':

    J = float(os.getenv("J", '3'))
    Nstep = int(os.getenv("Nstep", '3000'))
    print(f'J={J} \nNstep={Nstep}')

    hmc = HmcSampler(J=J, Nstep=Nstep)

    # Measure
    G_avg, G_std = hmc.measure()

    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), hmc.N_step, hmc.specifics, False)

    plt.show()

    exit()

