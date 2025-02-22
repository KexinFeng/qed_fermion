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

from qed_fermion.coupling_mat2 import initialize_coupling_mat

script_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

class HmcSampler(object):
    def __init__(self, config=None):
        self.Lx = 10
        self.Ly = 10
        self.Ltau = 20
        self.J = 1
        self.boson = None
        self.A = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, self.J)[0].to(device)
        self.bs = 5

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
        self.delta_t = 0.05
        
        # m = 2, T = 1.5, 
        # T/4 = 0.375

        total_t = 0.375
        self.N_leapfrog = int(total_t // self.delta_t)

        self.N_leapfrog = 7
        # Fixing the total number of leapfrog step, then the larger delta_t, the longer time the Hamiltonian dynamic will reach, the less correlated is the proposed config to the initial config, where the correlation is in the sense that, in the small delta_t limit, almost all accpeted and p being stochastic, then the larger the total_t, the less autocorrelation. But larger delta_t increases the error amp and decreases the acceptance rate.

        # Increasing m, say by 4, the sigma(p) increases by 2. omega = sqrt(k/m) slows down by 2 [cos(wt) ~ 1 - 1/2 * k/m * t^2, but ]. The S amplitude is not affected (since it's decided by initial cond.), but somehow H amplitude decreases by 4, similar to omega^2 decreases by 4. 

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False

    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau, device=device)
        # self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.1

        self.boson = torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * torch.linspace(0.5, 1, self.bs, device=device).view(-1, 1, 1, 1, 1)
       
    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [bs, 2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * math.sqrt(self.m)

    def force(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [bs, 2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x.
        """
        return -torch.einsum('ijklmnop,bmnop->bijkl', self.A, x)


    def leapfrog_proposer(self):
        """          
        Propose new boson according self.boson, which consists of traj_length steps. At each step the force will be evaluated.

        The action S = 1/2 * boson.T * self.A * boson + momentum**2.

        # Primitive
        x_0 = x
        p_{1/2} = p_0 + dt/2 * F(x_0)

        x_{n+1} = x_{n} + 2 * p_{n+1/2} dt
        p_{n+3/2} = p_{n+1/2} + F(x_{n+1}) dt 

        p_{N} = (p_{N+1/2} + p_{N-1/2}) /2

        :return: trial_boson, trial_momentum
        """

        p0 = self.draw_momentum()
        p = p0.clone()
        x = self.boson.clone()
        H0, S0 = self.action(p, x)
        dt = self.delta_t

        p_last = p
        p = p_last + dt /2 * self.force(x)
        
        if self.debug_pde:
            b_idx = 3
            
            # Initialize plot
            # plt.ion()  # Turn on interactive mode
            # fig, ax = plt.subplots()
            fig, axs = plt.subplots(2, 1, figsize=(6, 8))  # Two rows, one column

            Hs = [H0[b_idx].item()]
            Ss = [S0[b_idx].item()] 

            # Setup for first subplot (Hs)
            line_Hs, = axs[0].plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
            axs[0].set_ylabel('Hamiltonian (H)')
            axs[0].legend()
            axs[0].grid()

            # Setup for second subplot (Ss)
            # axs[1].set_title('Real-Time Evolution of S_s')
            line_Ss, = axs[1].plot(Ss, marker='s', linestyle='-', color='r', label='S_s')
            axs[1].set_xlabel('Leapfrog Step')
            axs[1].set_ylabel('S')
            axs[1].legend()
            axs[1].grid()

        leapfrog_step = self.N_leapfrog
        for _ in range(leapfrog_step):
            # x_{n+1} = x_{n} + 2 * p_{n+1/2} dt
            # p_{n+3/2} = p_{n+1/2} + F(x_{n+1}) dt  
            
            p_last = p
            x = x + p / self.m * dt      
            p = p + self.force(x) * dt

            if self.debug_pde:
                Hd, Sd = self.action((p + p_last)/2, x)  # Append new H value   
                Hs.append(Hd[b_idx].item())
                Ss.append(Sd[b_idx].item())

                # Update data for both subplots
                line_Hs.set_data(range(len(Hs)), Hs)
                line_Ss.set_data(range(len(Ss)), Ss)

                # Adjust limits dynamically
                axs[0].relim()
                axs[0].autoscale_view()
                amp = max(Hs) - min(Hs)
                axs[0].set_title(f'dt={self.delta_t:.2f}, m={self.m}, amp={amp:.2f}, N={leapfrog_step}')

                axs[1].relim()
                axs[1].autoscale_view()
                amp = max(Ss) - min(Ss)
                axs[1].set_title(f'dt={self.delta_t:.2f}, m={self.m}, amp={amp:.2f}, N={leapfrog_step}') 

                plt.pause(0.1)   # Small delay to update the plot

        return x, (p + p_last)/2, p0


    def action(self, momentum, boson):
        """
        The action S = 1/2 * boson.transpose * self.A * boson + momentum**2. The prob ~ e^{-S}.

        :param momentum: [bs, 2, Lx, Ly, Ltau] tensor
        :param boson: [bs, 2, Lx, Ly, Ltau] tensor
        :return: the action
        """
        kinetic = torch.sum(momentum ** 2, axis=(1, 2, 3, 4)) / (2 * self.m)
        potential = 0.5 * torch.einsum('bijkl,bijkl->b', boson, -self.force(boson))
        return kinetic + potential, potential

    def metropolis_update(self):
        """
        Perform one step of metropolis update. Update self.boson.

        H_new = np.sum(p**2) / 2 + self.action(phi)
        if np.random.rand() < np.exp(H_old - H_new):
            self.phi = phi  # Accept new configuration

        :return: None
        """
        # Given the last boson (conditional on the past) and momentum (iid sampled), the join dist. is the desired one. Then, the leapfrog proposes new config and the metropolis update preserves the join dist. The marginal dist. of the config is always conditional on the past while the momentum is not. Kinetic + potential (action) is conserved in the Hamiltonian dynamics but the action is not.

        boson_new, p_new, p_old = self.leapfrog_proposer()
        H_old, S_old = self.action(p_old, self.boson)
        H_new, S_new = self.action(p_new, boson_new)
        # print(f"H_old, H_new, diff: {H_old}, {H_new}, {H_new - H_old}")
        # print(f"threshold: {torch.exp(H_old - H_new).item()}")

        accp = torch.rand(self.bs, device=device) < torch.exp(H_old - H_new)
        self.boson[accp] = boson_new[accp]
        return self.boson, accp, S_old, S_new


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

    
    def measure(self):
        """
        boson: [2, Lx, Ly, Ltau]

        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        self.initialize_boson()
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
            boson, accp, Sold, Snew = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
            self.G_list[i] = accp.view(-1, 1) * self.curl_greens_function_batch(boson) + (1 - accp.view(-1, 1).to(torch.float)) * self.G_list[i-1]
            self.S_list[i] = torch.where(accp, Snew, Sold)
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

    start = 100
    end = step
    seq_idx = np.arange(start, end, 1)

    ## Plot
    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_avg', color='blue', lw=2)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$|G(\tau)|$")
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
    plt.ylabel('log10(|G|) values')
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
    # Lx, Ly, Ltau = 20, 20, 20
    step = 1100
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), step)

    plt.show()
    dbstop = 1
