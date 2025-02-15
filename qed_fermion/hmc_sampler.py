import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')

import os
import pickle

from qed_fermion.coupling_mat2 import initialize_coupling_mat


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

class HmcSampler(object):
    def __init__(self, config=None):
        self.Lx = 10
        self.Ly = 10
        self.Ltau = 10
        self.J = 1
        self.boson = None
        self.A = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, self.J).to(device)

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 20
        self.N_step = 2000
        self.step = 0
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.num_tau + 1)
        self.accp_list = torch.zeros(self.N_step, dtype=torch.bool)
        self.accp_rate = torch.zeros(self.N_step)
        self.S_list = torch.zeros(self.N_step + self.N_therm_step)
        self.H_list = torch.zeros(self.N_step + self.N_therm_step)

        # Leapfrog
        self.delta_t_thrm = 0.01
        self.delta_t = 0.1
        total_t = 0.5
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
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau)
        self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 1
        
    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device)

    def force(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x.
        """
        return -torch.einsum('ijklmnop,mnop->ijkl', self.A, x)

    def leapfrog_proposer(self):
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

        for i in range(self.N_leapfrog_thrm if self.thrm_bool else self.N_leapfrog):
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

    def action(self, momentum, boson):
        """
        The action S = 1/2 * boson.transpose * self.A * boson + momentum**2. The prob ~ e^{-S}.

        :param momentum: [2, Lx, Ly, Ltau] tensor
        :param boson: [2, Lx, Ly, Ltau] tensor
        :return: the action
        """
        kinetic = torch.sum(momentum ** 2)
        potential = 1/2 * torch.einsum('ijkl,ijkl->', boson, -self.force(boson))
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
        print(f"H_old, H_new, diff: {H_old}, {H_new}, {H_old - H_new}")
        print(f"threshold: {torch.exp(H_old - H_new).item()}")
        accp =  torch.rand(1, device=device) < torch.exp(H_old - H_new)
        print(accp.item())
        if accp:
            self.boson = boson_new
            return self.boson, True, H_new, S_new
        else:
            return self.boson, False, H_old, S_old

    def greens_function(self, boson):
        """
        Evaluate the greens function of boson.
        obsrv = phi^{a}_r1 phi^{a}_r2.
        r = (x, y, tau), x1 = x2, y1 = y2, tau2 = tau1 + dtau
        obsrv = [mean(phi[a, :, :, :_taus] * phi[a, :, :, :_taus + dtau], axis = [0, 1, 2]) for dtau in range(0, num_dtau)]

        :param boson: [2, Lx, Ly, Ltau] tensor
        :return: a vector of shape [num_dtau]
        """
        correlations = []
        boson_elem = boson[self.polar, 0, 0]
        for dtau in range(self.num_tau + 1):
            idx1 = list(range(self.Ltau))
            idx2 = [(i+dtau) % self.Ltau for i in idx1]
            corr = torch.mean(boson_elem[..., idx1] * boson_elem[..., idx2], dim=(0))
            correlations.append(corr)

        return torch.stack(correlations)
    
    def measure(self):
        """
        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        self.initialize_boson()

        # Thermalize
        self.thrm_bool = True
        # boson = None
        for i in range(self.N_therm_step):
            boson, accp, H, S = self.metropolis_update()
            self.H_list[i] = H
            self.S_list[i] = S

        self.G_list[-1] = self.greens_function(boson)

        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp, H, S = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float)).item()
            self.G_list[i] = self.greens_function(boson) if accp else self.G_list[i-1]
            self.H_list[i + self.N_therm_step] = H
            self.S_list[i + self.N_therm_step] = S

        res = {'mean': self.G_list.mean(dim=0),
               'std': self.G_list.std(dim=0)}

        # Save to file
        data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/data_hmc/"
        file_name = f"corr_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_tau-max_{self.num_tau}"
        self.save_to_file(res, data_folder, file_name)

        return res['mean'], res['std']

    # ------- Save to file -------
    def save_to_file(self, res, data_folder, filename):
        os.makedirs(data_folder, exist_ok=True)
        filepath = os.path.join(data_folder, f"{filename}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(res, f)
        print(f"Data saved to {data_folder + filepath}")

    # ------- Visualization -------
    def total_monitoring(self):
        """
        Visualize obsrv and accp in subplots.
        """
        plt.figure()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[2].plot(self.accp_rate.numpy())
        axes[2].set_xlabel("Steps")
        axes[2].set_ylabel("Acceptance Rate")

        idx = [0, self.num_tau // 2, -2]
        axes[1].plot(self.G_list[:, idx[0]].numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[:, idx[1]].numpy(), label=f'G[{self.num_tau // 2}]')
        axes[1].plot(self.G_list[:, idx[2]].numpy(), label=f'G[-2]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

        # axes[2].plot(self.H_list[self.N_therm_step:].numpy())
        # axes[2].set_ylabel("H")

        axes[0].plot(self.S_list[self.N_therm_step:].numpy())
        axes[0].set_ylabel("S")


        plt.tight_layout()
        # plt.show()

    @staticmethod
    def visualize_final_greens(G_avg):
        """
        Visualize green functions with error bar
        """
        plt.figure()
        plt.plot(G_avg.numpy())
        plt.xlabel("dtau")
        plt.ylabel("Greens Function")
        # plt.show()


if __name__ == '__main__':

    hmc = HmcSampler()
    G_avg, G_std = hmc.measure()

    # print(hmc.accp_list)

    # plt.figure()
    # # hmc.visualize_final_greens(G_avg)
    # plt.plot(hmc.accp_list.numpy())
    # plt.xlabel("Steps")
    # plt.ylabel("Acceptance Rate")
    # plt.show()

    hmc.total_monitoring()
    hmc.visualize_final_greens(G_avg)

    plt.show()
    dbstop = 1
