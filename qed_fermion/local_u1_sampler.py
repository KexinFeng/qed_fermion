import os
import time
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
script_path = os.path.dirname(os.path.abspath(__file__))

from qed_fermion.coupling_mat2 import initialize_coupling_mat
from qed_fermion.hmc_sampler import HmcSampler
import math
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"device: {device}")

class LocalU1Sampler(HmcSampler):
    def __init__(self, N_step=int(1000 * 100), config=None):
        self.Lx = 20
        self.Ly = 20
        self.Ltau = 20
        self.J = 1
        self.K = 1
        self.boson = None
        self.A, self.unit = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, J=self.J, K=self.K)
        self.A = self.A.to(device)
        self.unit = self.unit.to(device)

        self.num_site = self.Lx * self.Ly * self.Ltau

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 1000
        swp = 100
        self.N_step = self.num_site * swp
        self.step = 0
        self.cur_step = 0
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.num_tau + 1)
        self.accp_list = torch.zeros(self.N_step, dtype=torch.bool)
        self.accp_rate = torch.zeros(self.N_step)
        self.S_list = torch.zeros(math.ceil(self.N_step / (self.Lx * self.Ly * self.Ltau)))

        # Leapfrog
        self.delta_t_thrm = 0.01
        self.delta_t = 0.1
        total_t = 0.5
        self.N_leapfrog_thrm = int(total_t // self.delta_t_thrm)
        self.N_leapfrog = int(total_t // self.delta_t)
        # self.N_leapfrog = 20
        self.w = 0.5 * torch.pi
        self.w_thrm = 0.05 * torch.pi

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False
        self.debug = True

    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.ones(2, self.Lx, self.Ly, self.Ltau, device=device)
        # self.boson[0] = -1
        self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.42

    def force(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x.
        """
        return -torch.einsum('ijklmnop,mnop->ijkl', self.A, x)

    # def action(self, boson=None):
    #     """
    #     The action S = 1/2 * boson.transpose * self.A * boson.

    #     :param boson: [2, Lx, Ly, Ltau] tensor
    #     :return: the action
    #     """
    #     if boson is None:
    #         boson = self.boson
    #     potential = 0.5 * torch.einsum('ijkl,ijkl->', boson, -self.force(boson))
    #     return potential
    
    def action(self, boson=None):
        A = self.A.permute([3, 2, 1, 0, 7, 6, 5, 4]).reshape(self.num_site*2, self.num_site*2)
        if boson is None:
            boson = self.boson
        # potential = 0.5 * torch.einsum('ijkl,ijkl->', boson, -self.force(boson))

        boson2 = boson.permute([3, 2, 1, 0]).reshape(-1)
        potential2 = 0.5 * boson2 @ A @ boson2

        # torch.testing.assert_close(potential, potential2)
        return potential2
   
    def tau_link(self, x, y, tau, boson):
        return 0.5 * self.J * torch.sum( (boson[:, x, y, (tau+1) % self.Ltau] - boson[:, x, y, tau]) ** 2, axis=0  )
    
    def plaq(self, x, y, tau, boson):
        return 0.5 * self.K * (boson[0, x, y, tau] - boson[0, x, (y+1)%self.Ly, tau] + boson[1, (x+1)%self.Lx, y, tau] - boson[1, x, y, tau]) ** 2
    
    def d_action(self, boson_new, site):
        """
        :param boson_new: [2, Lx, Ly, Ltau] tensor
        :param site: (x, y, tau)
        :return:  S_new - S_old
        """
        x, y, tau = site
        S2_new, S2_old = 0, 0
        S1_new, S1_old = 0, 0

        # tau_link = lambda x, y, tau, phi: 0.5 * self.J * torch.sum( (phi[:, x, y, (tau+1) % self.Ltau] - phi[:, x, y, tau]) ** 2, axis=0  )
        S1_new = self.tau_link(x, y, tau, boson_new) + self.tau_link(x, y, (tau-1) % self.Ltau, boson_new)
        S1_old = self.tau_link(x, y, tau, self.boson) + self.tau_link(x, y, (tau-1) % self.Ltau, self.boson)

        # plaq = lambda x, y, tau, phi: 0.5 * (phi[0, x, y, tau] - phi[0, x, (y+1)%self.Ly, tau] + phi[1, (x+1)%self.Lx, y, tau] - phi[1, x, y, tau]) ** 2
        S2_new = self.plaq(x, y, tau, boson_new) \
                + self.plaq((x-1)%self.Lx, y, tau, boson_new) \
                + self.plaq(x, (y-1)%self.Ly, tau, boson_new) 
                # + self.plaq((x-1)%self.Lx, (y-1)%self.Ly, tau, boson_new)      
        S2_old = self.plaq(x, y, tau, self.boson) \
                + self.plaq((x-1)%self.Lx, y, tau, self.boson) \
                + self.plaq(x, (y-1)%self.Ly, tau, self.boson) 
                # + self.plaq((x-1)%self.Lx, (y-1)%self.Ly, tau, self.boson)

        # torch.testing.assert_close(self.plaq((x-1)%self.Lx, (y-1)%self.Ly, tau, boson_new),     
        #                            self.plaq((x-1)%self.Lx, (y-1)%self.Ly, tau, self.boson)
        # )
        
        # s = 0
        # ss = 0
        # for k in range(self.Ltau):
        #     for j in range(self.Ly):
        #         for i in range(self.Lx):
        #             s += self.plaq(i, j, k, boson_new)

        #             dim = self.unit.size(0)
        #             lindex = lambda a, i, j, k: a + i * 2 + j * 2*self.Lx + k * 2*self.Lx*self.Ly
        #             idxs = torch.arange(lindex(0, i, j, k), lindex(0, i, j, k) + dim) % (self.num_site * 2)
        #             boson_part = boson_new.permute([3, 2, 1, 0]).reshape(-1)[idxs]
        #             ss += 0.5 * self.K * boson_part[:5] @ self.unit[:5, :5] @ boson_part[:5].T
                    
        #             torch.testing.assert_close(s, ss)

        # s2 = self.action(boson_new)
        # torch.testing.assert_close(s, s2)
        
        return (S2_new - S2_old + S1_new - S1_old)[0]

    def local_u1_proposer_per_layer(self):
        """
        Propose a new boson configuration by modifying a single tau-layer.

        The new boson values are sampled within a window of size `self.w` centered at the current value,
        with periodic boundary conditions ensuring boson ∈ [-π, π] mod 2π.

        :return: trial_boson
        """
        boson_new = self.boson.clone()

        # Select tau index based on the current step
        idx = self.step % self.Ltau

        # Sample new values in the window [boson - w, boson + w], with periodicity mod 2π
        delta = (torch.rand_like(boson_new[:, :, :, idx]) - 0.5) * 2 * self.w  # Uniform in [-w, w]
        boson_new[:, :, :, idx] += delta

        # Apply periodic boundary condition: ensure values remain in [-π, π] mod 2π
        boson_new[:, :, :, idx] = (boson_new[:, :, :, idx] + torch.pi) % (2 * torch.pi) - torch.pi

        return boson_new
    
    
    def local_u1_proposer_per_site(self):
        """
        Propose a new boson configuration by modifying a single site.

        The new boson values are sampled within a window of size `self.w` centered at the current value,
        with periodic boundary conditions ensuring boson ∈ [-π, π] mod 2π.

        :return: trial_boson
        """
        boson_new = self.boson.clone()

        win_size = self.w if not self.thrm_bool else self.w_thrm

        # Select tau index based on the current step
        idx_x, idx_y, idx_tau = torch.unravel_index(torch.tensor([self.step], device=boson_new.device), (self.Lx, self.Ly, self.Ltau))
        # idx_x = idx_x.item()
        # idx_y = idx_y.item()
        # idx_tau = idx_tau.item()

        # # Sample new values in the window [boson - w, boson + w], with periodicity mod 2π
        # delta = (torch.rand_like(boson_new[:, idx_x, idx_y, idx_tau]) - 0.5) * 2 * win_size # Uniform in [-w, w]
        # boson_new[:, idx_x, idx_y, idx_tau] += delta

        # # Apply periodic boundary condition: ensure values remain in [-π, π] mod 2π
        # boson_new[:, idx_x, idx_y, idx_tau] = (boson_new[:, idx_x, idx_y, idx_tau] + torch.pi) % (2 * torch.pi) - torch.pi

        # Sample new values for x, y components on site (idx_x, idx_y, idx_tau) in the window [boson - w, boson + w]
        delta = (torch.rand_like(boson_new[:, idx_x, idx_y, idx_tau]) - 0.5) * 2 * win_size # Uniform in [-w/2, w/2]
        boson_new[:, idx_x, idx_y, idx_tau] += delta

        return boson_new, (idx_x, idx_y, idx_tau)
    
    
    def local_u1_proposer(self):
        """
        Propose a new boson configuration by modifying a single tau-layer.

        The new boson values are sampled within a window of size `self.w` centered at the current value,
        with periodic boundary conditions ensuring boson ∈ [-π, π] mod 2π.

        :return: trial_boson
        """
        boson_new = self.boson.clone()

        # # Select tau index based on the current step
        # idx = self.step % self.Ltau

        # Sample new values in the window [boson - w, boson + w], with periodicity mod 2π
        delta = (torch.rand_like(boson_new[:, :, :, :]) - 0.5) * 2 * self.w  # Uniform in [-w, w]
        boson_new[:, :, :, :] += delta

        # Apply periodic boundary condition: ensure values remain in [-π, π] mod 2π
        boson_new[:, :, :, :] = (boson_new[:, :, :, :] + torch.pi) % (2 * torch.pi) - torch.pi

        return boson_new

    def metropolis_update(self):
        """
        Perform a single Metropolis update using local U(1) updates.

        :return: Updated boson configuration, acceptance flag, new action
        """
        boson_new, site = self.local_u1_proposer_per_site()
        # S_old = self.action(self.boson)
        # S_new = self.action(boson_new)
        # d_action0 = self.action(boson_new) - self.action(self.boson)
        d_action = self.d_action(boson_new, site)
        
        # torch.testing.assert_close(d_action, d_action0, atol=5e-4, rtol=1e-3)

        accp =  torch.rand(1, device=device) < torch.exp(-d_action)
        if self.debug and self.thrm_bool:
            # print(f"diff: {S_old - S_new}")
            print(f"diff: {d_action}")
            print(f"threshold: {torch.exp(d_action).item()}")
            print(accp.item())

        if accp:
            self.boson = boson_new
            return self.boson, True
        else:
            return self.boson, False

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
        boson: [2, Lx, Ly, Ltau]
        
        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        self.initialize_boson()

        # # Thermalize
        # self.thrm_bool = True
        # for i in range(self.N_therm_step):
        #     boson, accp, S = self.metropolis_update()
        #     self.S_list[i] = S
        #     self.step += 1

        self.G_list[-1] = self.greens_function(self.boson)

        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float)).item()
            self.G_list[i] = self.greens_function(boson) if accp else self.G_list[i-1]
            if i % self.num_site == 0:
                i_swp = i // self.num_site
                self.S_list[i_swp] = self.action()
            self.step += 1
            self.cur_step += 1

            # plotting
            if i % (self.num_site * 10) == 0:
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)  # Pause for 5 seconds
                plt.close()

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)

        res = {'boson': boson,
               'step': self.step,
               'wsz': self.w,
               'G_avg': G_avg,
               'G_std': G_std,
               'G_list': self.G_list}

        # # Save to file
        # data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/data_local_u1/"
        # file_name = f"corr_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_tau-max_{self.num_tau}"
        # self.save_to_file(res, data_folder, file_name)

        # Save check point
        data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/"
        file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
        self.save_to_file(res, data_folder, file_name)

        return G_avg, G_std
    
    def resume_measure(self):
        data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/"
        file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
        file_path = os.path.join(data_folder, f"{file_name}.pt")
        res = torch.load(file_path)
        self.step = res['step']
        self.boson = res['boson']
        self.G_list[-1] = self.greens_function(self.boson)

        print(f'Energy: {self.action(self.boson)}')

        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float)).item()
            self.G_list[i] = self.greens_function(boson) if accp else self.G_list[i-1]
            if i % (self.Lx * self.Ly * self.Ltau) == 0:
                i_swp = i // (self.Lx * self.Ly * self.Ltau)
                self.S_list[i_swp] = self.action()
            self.step += 1
            self.cur_step += 1

            # plotting
            if i % (self.Lx * self.Ly * self.Ltau) == 0:
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)  # Pause for 5 seconds
                plt.close()

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'wsz': self.w,
               'G_avg': G_avg,
               'G_std': G_std,
               'G_list': self.G_list}
        
        print(f'Energy: {self.action(self.boson)}')
        # Save check point
        file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
        self.save_to_file(res, data_folder, file_name)

        return G_avg, G_std

    def total_monitoring(self):
        """
        Visualize obsrv and accp in subplots.
        """
        # plt.figure()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[2].plot(self.accp_rate[:self.cur_step].numpy())
        axes[2].set_xlabel("Steps")
        axes[2].set_ylabel("Acceptance Rate")

        idx = [0, self.num_tau // 2, -2]
        axes[1].plot(self.G_list[: self.cur_step, idx[0]].numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[: self.cur_step, idx[1]].numpy(), label=f'G[{self.num_tau // 2}]')
        axes[1].plot(self.G_list[: self.cur_step, idx[2]].numpy(), label=f'G[-2]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

        # axes[2].plot(self.H_list[self.N_therm_step:].numpy())
        # axes[2].set_ylabel("H")
        idx_swp = self.cur_step // self.num_site
        axes[0].plot(self.S_list[:idx_swp + 1].numpy())
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

    # ------- Visualization -------
    def visualize_final_greens(self, G_avg, G_std):
        """
        Visualize green functions with error bar
        """
        plt.figure()
        plt.errorbar(x=list(range(len(G_avg))), y=G_avg.numpy(), yerr=G_std.numpy(), linestyle='-', marker='o', )
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$|G(\tau)|$")

        class_name = self.__class__.__name__
        method_name = "greens"
        save_dir = os.path.join(script_path, f"figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")
        
    def visualize_final_greens_loglog(self, G_avg, G_std):
        """
        Visualize green functions with error bar
        """
        x=np.array(list(range(len(G_avg))))
        G_avg = G_avg.numpy()

        plt.figure()
        plt.plot(np.log10(x+1), np.log10(abs(G_avg)), linestyle='-', marker='o', label='log|G_avg|', color='blue', lw=2)
        # plt.errorbar((x+1), G_avg, yerr=G_std, linestyle='-', marker='o', label='log|G_avg|', color='blue', lw=2)
        # plt.xscale('log')
        # plt.yscale('log')

        # Add labels and title
        plt.xlabel('X-axis label')
        plt.ylabel('log10(|G|) values')
        plt.title('Log-Log Plot of G_avg')
        plt.legend()


        xs, ys = np.log10(x+1), np.log10(abs(G_avg))
        mid_idx = len(xs) // 2

        # Get the coordinates of the 0th and midpoint
        x0, y0 = xs[1], ys[1]
        x_mid, y_mid = xs[mid_idx], ys[mid_idx]

        # Compute the slope
        slope = (y_mid - y0) / (x_mid - x0)

        # Plot the points and the connecting line
        plt.plot(xs, ys, 'bo-', label="Data")
        plt.plot([x0, x_mid], [y0, y_mid], 'r-', linewidth=2, label=f"Line (slope={slope:.2f})")

        # Annotate the slope
        mid_x_pos = (x0 + x_mid) / 2
        mid_y_pos = (y0 + y_mid) / 2
        plt.annotate(f"Slope = {slope:.2f}", (mid_x_pos, mid_y_pos), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=12, color='red')

        # save_plot
        class_name = self.__class__.__name__
        method_name = "greens_loglog"
        save_dir = os.path.join(script_path, f"figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")


if __name__ == '__main__':

    hmc = LocalU1Sampler()

    G_avg, G_std = hmc.measure()
    hmc.total_monitoring()
    hmc.visualize_final_greens(G_avg, G_std)
    hmc.visualize_final_greens_loglog(G_avg, G_std)

    # hmc.step = 10000
    # G_avg, G_std = hmc.resume_measure()

    # hmc.total_monitoring()
    # hmc.visualize_final_greens(G_avg, G_std)
    # hmc.visualize_final_greens_loglog(G_avg, G_std)

    plt.show(block=True)
    dbstop = 1
