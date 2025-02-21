import json
import math
import matplotlib
import numpy as np
matplotlib.use('MacOSX')

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')

import os
import pickle

from qed_fermion.coupling_mat2 import initialize_coupling_mat

script_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

class HmcSampler(object):
    def __init__(self, config=None):
        self.Lx = 20
        self.Ly = 20
        self.Ltau = 20
        self.J = 1
        self.boson = None
        self.A = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, self.J)[0].to(device)

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 0
        self.N_step = int(1e3) + 100
        self.step = 0
        self.cur_step = 0
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.num_tau + 1)
        self.accp_list = torch.zeros(self.N_step, dtype=torch.bool)
        self.accp_rate = torch.zeros(self.N_step)
        self.S_list = torch.zeros(self.N_therm_step + self.N_step)
        self.H_list = torch.zeros(self.N_therm_step + self.N_step)

        # Leapfrog
        self.m = 1/2 * 4
        self.delta_t_thrm = 0.05
        self.delta_t = 0.05
        total_t = 1
        self.N_leapfrog_thrm = int(total_t // self.delta_t_thrm)
        self.N_leapfrog = int(total_t // self.delta_t)
        # self.N_leapfrog = 20

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False

    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau, device=device)
        self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.001
        
    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 1/math.sqrt(2)

    def force(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x.
        """
        return -torch.einsum('ijklmnop,mnop->ijkl', self.A, x)

    def leapfrog_proposer_2nd(self):
        """          
        Propose new boson according self.boson, which consists of traj_length steps. At each step the force will be evaluated.

        The action S = 1/2 * boson.T * self.A * boson + momentum**2. The prob ~ e^{-S}.
        x_{n+1/2} = x_{n} + 2p_{n} dt/2
        p_{n+1} = p_{n} + F(x_{n+1/2}) dt
        x_{n+1} = x_{n+1/2} + 2p_{n+1} dt/2

        :return: trial_boson, trial_momentum
        """

        p0 = self.draw_momentum()
        p = p0.clone()
        x = self.boson.clone()
        H0 = self.action(p, x)[0].item()
        dt = self.delta_t_thrm if self.thrm_bool else self.delta_t
        
        if self.debug_pde:
            # Initialize plot
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            Hs = [H0]

            # Plot setup
            line, = ax.plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
            ax.set_xlabel('Leapfrog Step')
            ax.set_ylabel('Hamiltonian (H)')
            ax.set_title('Real-Time Evolution of H_s')
            ax.legend()
            plt.grid()

        leapfrog_step = self.N_leapfrog_thrm if self.thrm_bool else self.N_leapfrog
        for _ in range(leapfrog_step):
            x = x + dt * p
            p = p + dt * self.force(x)
            x = x + dt * p

            if self.debug_pde:
                Hs.append(self.action(p, x)[0].item())  # Append new H value

                # Update plot
                line.set_ydata(Hs)
                line.set_xdata(range(len(Hs)))
                ax.relim()  # Recalculate limits
                ax.autoscale_view()  # Rescale view

                plt.draw()
                plt.pause(0.01)  # Pause for smooth animation

        return x, p, p0


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

        # 2nd order
        x_{n+1/2} = x_{n} + 2p_{n} dt/2
        p_{n+1} = p_{n} + F(x_{n+1/2}) dt
        x_{n+1} = x_{n+1/2} + 2p_{n+1} dt/2

        :return: trial_boson, trial_momentum
        """

        p0 = self.draw_momentum()
        p = p0.clone()
        x = self.boson.clone()
        H0 = self.action(p, x)[0].item()
        dt = self.delta_t_thrm if self.thrm_bool else self.delta_t

        p_last = p
        p = p_last + dt /2 * self.force(x)
        
        if self.debug_pde:
            # Initialize plot
            # plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            Hs = [H0]

            # Plot setup
            line, = ax.plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
            ax.set_xlabel('Leapfrog Step')
            ax.set_ylabel('Hamiltonian (H)')
            ax.set_title('Real-Time Evolution of H_s')
            ax.legend()
            plt.grid()

        leapfrog_step = self.N_leapfrog_thrm if self.thrm_bool else self.N_leapfrog
        for _ in range(leapfrog_step):
            # x_{n+1} = x_{n} + 2 * p_{n+1/2} dt
            # p_{n+3/2} = p_{n+1/2} + F(x_{n+1}) dt  
            
            p_last = p
            x = x + p / self.m * dt      
            p = p + self.force(x) * dt

            # x = x + dt * p
            # p = p + dt * self.force(x)
            # x = x + dt * p

            if self.debug_pde:
                Hs.append(self.action((p + p_last)/2, x)[0].item())  # Append new H value

                # Update plot
                line.set_ydata(Hs)
                line.set_xdata(range(len(Hs)))
                ax.relim()  # Recalculate limits
                ax.autoscale_view()  # Rescale view

                plt.draw()
                plt.pause(0.01)  # Pause for smooth animation

        return x, (p + p_last)/2, p0


    def action(self, momentum, boson):
        """
        The action S = 1/2 * boson.transpose * self.A * boson + momentum**2. The prob ~ e^{-S}.

        :param momentum: [2, Lx, Ly, Ltau] tensor
        :param boson: [2, Lx, Ly, Ltau] tensor
        :return: the action
        """
        kinetic = torch.sum(momentum ** 2) / (2 * self.m)
        potential = 0.5 * torch.einsum('ijkl,ijkl->', boson, -self.force(boson))
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

        accp =  torch.rand(1, device=device) < torch.exp(H_old - H_new)
        if accp:
            self.boson = boson_new
            return self.boson, True, H_new, S_new
        else:
            return self.boson, False, H_old, S_old


    def curl_greens_function_batch(self, boson):
        """
        Evaluate the Green's function of the curl of boson field.
        obsrv = curl(phi) at (x, y, tau1) * curl(phi) at (x, y, tau2),
        summed over (x, y) and averaged over the batch.

        :param boson: [batch_size, 2, Lx, Ly, Ltau] tensor
        :return: a tensor of shape [bs, num_dtau]
        """

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
        self.G_list[-1] = self.curl_greens_function_batch(self.boson).unsqueeze(0)

        # Thermalize
        self.thrm_bool = True
        # boson = None
        for i in range(self.N_therm_step):
            boson, accp, H, S = self.metropolis_update()
            self.H_list[i] = H
            self.S_list[i] = S
        self.G_list[-1] = self.curl_greens_function_batch(boson).unsqueeze(0)

        plt.figure()
        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp, H, S = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float)).item()
            self.G_list[i] = self.curl_greens_function_batch(boson).unsqueeze(0) if accp else self.G_list[i-1]
            self.H_list[i + self.N_therm_step] = H
            self.S_list[i + self.N_therm_step] = S

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
                        'G_list': self.G_list}
                
                data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/check_point/"
                file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
                self.save_to_file(res, data_folder, file_name)           

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)

        res = {'boson': boson,
                'step': self.step,
                'G_list': self.G_list}

        # Save to file
        data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/check_point/"
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
    def total_monitoring_batch(self):
        """
        Visualize obsrv and accp in subplots.
        """
        # plt.figure()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        for b in range(self.accp_list.size(1)):
            axes[2].plot(self.accp_rate[:self.cur_step, b].cpu().numpy())
        axes[2].set_xlabel("Steps")
        axes[2].set_ylabel("Acceptance Rate")

        idx = [0, self.num_tau // 2, -2]
        # for b in range(self.G_list.size(1)):
        axes[1].plot(self.G_list[:self.cur_step, ..., idx[0]].mean(axis=1).cpu().numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[:self.cur_step, ..., idx[1]].mean(axis=1).cpu().numpy(), label=f'G[{self.num_tau // 2}]')
        axes[1].plot(self.G_list[:self.cur_step, ..., idx[2]].mean(axis=1).cpu().numpy(), label=f'G[-2]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

        # axes[2].plot(self.H_list[self.N_therm_step:].cpu().numpy())
        # axes[2].set_ylabel("H")
        idx_swp = self.cur_step // self.num_site
        for b in range(self.S_list.size(1)):
            axes[0].plot(self.S_list[:idx_swp, b].cpu().numpy(), 'o')
        axes[0].set_ylabel("S")

        plt.tight_layout()
        # plt.show()

        class_name = self.__class__.__name__
        method_name = "totol_monit"
        save_dir = os.path.join(script_path, f"figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.png")
        plt.savefig(file_path)
        print(f"Figure saved at: {file_path}")


    def total_monitoring(self):
        """
        Visualize obsrv and accp in subplots.
        """
        # plt.figure()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        

        axes[2].plot(self.accp_rate[:self.cur_step].cpu().numpy())
        axes[2].set_xlabel("Steps")
        axes[2].set_ylabel("Acceptance Rate")

        idx = [0, self.num_tau // 2, -2]
        # for b in range(self.G_list.size(1)):
        axes[1].plot(self.G_list[:self.cur_step, ..., idx[0]].cpu().numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[:self.cur_step, ..., idx[1]].cpu().numpy(), label=f'G[{self.num_tau // 2}]')
        axes[1].plot(self.G_list[:self.cur_step, ..., idx[2]].cpu().numpy(), label=f'G[-2]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

        # axes[2].plot(self.H_list[self.N_therm_step:].cpu().numpy())
        # axes[2].set_ylabel("H")
        axes[0].plot(self.S_list[self.N_therm_step: self.N_therm_step + self.cur_step].cpu().numpy(), 'o')
        axes[0].set_ylabel("S")

        plt.tight_layout()
        # plt.show(block=False)

        class_name = self.__class__.__name__
        method_name = "totol_monit"
        save_dir = os.path.join(script_path, f"./figures/figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.png")
        plt.savefig(file_path)
        print(f"Figure saved at: {file_path}")


def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), step=1000001):
    """
    Visualize green functions with error bar
    """
    # Load numerical data
    data_folder_num = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/"

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    res = torch.load(filename)

    G_list = res['G_list']
    step = res['step']
    x = np.array(list(range(G_list[0].size(-1))))
    start, end = 0, step
    seq_idx = np.arange(start, end, 1)


    ## Plot
    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=0), yerr=G_list[seq_idx].numpy().std(axis=0)/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_avg', color='blue', lw=2)

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
    plt.errorbar(x+1, G_list[seq_idx].numpy().mean(axis=0), yerr=G_list[seq_idx].numpy().std(axis=0)/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx].numpy().mean(axis=0)

    # Analytical data
    data_folder_anal = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
    name = f"N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{Ltau}_num_{Ltau}"
    pkl_filepath = os.path.join(data_folder_anal, f"{name}.pkl")

    with open(pkl_filepath, 'r') as f:
        loaded_data_anal = json.load(f)

    G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1]
    taus = np.array(loaded_data_anal['taus'])  # The last element is the same as the first one due to the tau periodic boundary, and is thus removed.

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

    # hmc = HmcSampler()

    # G_avg, G_std = hmc.measure()

    # Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    Lx, Ly, Ltau = 20, 20, 20
    step = 1000
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), step)

    plt.show()
    dbstop = 1
