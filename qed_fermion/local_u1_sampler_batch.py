import json
import os
import time
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
script_path = os.path.dirname(os.path.abspath(__file__))

from qed_fermion.coupling_mat2 import initialize_coupling_mat
from qed_fermion.deprecated.hmc_sampler import HmcSampler
import math
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"device: {device}")

class LocalU1Sampler(HmcSampler):
    def __init__(self, N_step=int(1000 * 100), config=None):
        self.Lx = 10
        self.Ly = 10
        self.Ltau = 20
        self.J = 1
        self.K = 1
        self.boson = None
        self.A, self.unit = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, J=self.J, K=self.K)
        self.A = self.A.to(device)
        # self.unit = self.unit.to(device)
        self.bs = 5

        self.num_site = self.Lx * self.Ly * self.Ltau

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 10
        swp = int(800)
        self.N_step = self.num_site * swp
        self.step = 0
        self.cur_step = 0
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.bs, self.num_tau + 1, device=device)
        self.accp_list = torch.zeros(self.N_step, self.bs, dtype=torch.bool, device=device)
        self.accp_rate = torch.zeros(self.N_step, self.bs, device=device)
        self.S_list = torch.zeros(math.ceil(self.N_step / (self.Lx * self.Ly * self.Ltau)), self.bs, device=device)

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
        # self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.42
        self.boson = torch.randn(self.bs, 2, self.Lx, self.Ly, self.Ltau, device=device) * torch.linspace(0, 1, self.bs, device=device).view(-1, 1, 1, 1, 1)

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

        potential2 = []
        for b in range(self.bs):
            boson2 = boson[b].permute([3, 2, 1, 0]).reshape(-1)
            potential2.append(0.5 * boson2 @ A @ boson2)

        # torch.testing.assert_close(potential, potential2)
        return torch.stack(potential2)
   
    def tau_link(self, x, y, tau, boson):
        return 0.5 * self.J * torch.sum( (boson[..., x, y, (tau+1) % self.Ltau] - boson[..., x, y, tau]) ** 2, axis=-2  )
    
    def plaq(self, x, y, tau, boson):
        return 0.5 * self.K * (boson[..., 0, x, y, tau] - boson[..., 0, x, (y+1)%self.Ly, tau] + boson[..., 1, (x+1)%self.Lx, y, tau] - boson[..., 1, x, y, tau]) ** 2
    
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
   
        S2_old = self.plaq(x, y, tau, self.boson) \
                + self.plaq((x-1)%self.Lx, y, tau, self.boson) \
                + self.plaq(x, (y-1)%self.Ly, tau, self.boson) 

        return (S2_new - S2_old + S1_new - S1_old)[..., 0]

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

        # Sample new values for x, y components on site (idx_x, idx_y, idx_tau) in the window [boson - w, boson + w]
        delta = (torch.rand_like(boson_new[..., idx_x, idx_y, idx_tau]) - 0.5) * 2 * win_size # Uniform in [-w/2, w/2]
        boson_new[..., idx_x, idx_y, idx_tau] += delta

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

        accp =  torch.rand(self.bs, device=device) < torch.exp(-d_action)
        if self.debug and self.thrm_bool:
            # print(f"diff: {S_old - S_new}")
            print(f"diff: {d_action}")
            print(f"threshold: {torch.exp(d_action).item()}")
            print(accp.item())

        # if accp:
        #     self.boson = boson_new
        #     return self.boson, True
        # else:
        #     return self.boson, False

        self.boson[accp] = boson_new[accp]
        return self.boson, accp

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

    def curl_greens_function_batch(self, boson):
        """
        Evaluate the Green's function of the curl of boson field.
        obsrv = curl(phi) at (x, y, tau1) * curl(phi) at (x, y, tau2),
        summed over (x, y) and averaged over the batch.

        :param boson: [batch_size, 2, Lx, Ly, Ltau] tensor
        :return: a tensor of shape [bs, num_dtau]
        """
        batch_size, _, Lx, Ly, Ltau = boson.shape
        
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

        return torch.stack(correlations).T  # Shape: [num_dtau, batch_size]


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

        self.G_list[-1] = self.curl_greens_function_batch(self.boson)

        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float), axis=0)
            self.G_list[i] = accp.view(-1, 1) * self.curl_greens_function_batch(boson) + (1 - accp.view(-1, 1).to(torch.float)) * self.G_list[i-1]
            if i % self.num_site == 0:
                i_swp = i // self.num_site
                self.S_list[i_swp] = self.action()
            self.step += 1
            self.cur_step += 1

            # plotting
            if i % (self.num_site * 100) == 0:
                self.total_monitoring()
                plt.show(block=False)
                plt.pause(0.1)  # Pause for 5 seconds
                plt.close()

            # checkpointing
            if i % (self.num_site * 100) == 0:
                res = {'boson': boson,
                        'step': self.step,
                        'wsz': self.w,
                        'G_list': self.G_list.cpu()}
                
                data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/check_point/"
                file_name = f"ckpt_N_{self.Ltau}_Nx_{self.Lx}_Ny_{self.Ly}_step_{self.step}"
                self.save_to_file(res, data_folder, file_name)           

        G_avg, G_std = self.G_list.mean(dim=0), self.G_list.std(dim=0)
        res = {'boson': boson,
               'step': self.step,
               'wsz': self.w,
               'G_avg': G_avg,
               'G_std': G_std,
               'G_list': self.G_list.cpu()}

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
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")

    # ------- Visualization -------
    def visualize_final_greens(self, G_avg, G_std):
        """
        Visualize green functions with error bar
        """
        plt.figure()
        plt.errorbar(x=list(range(G_avg.size(-1))), y=G_avg.mean(axis=0).numpy(), yerr=G_std.mean(axis=0).numpy(), linestyle='-', marker='o', )
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$|G(\tau)|$")
        # plt.title(r"Ntau=20 Nx=20 Ny=20 Nswp=750")

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
        x=np.array(list(range(G_avg.size(-1))))
        # G_avg = G_avg.numpy()

        seq_idx = torch.arange(100 * self.num_site, 750 * self.num_site, self.num_site)
        # x=np.array(list(range(G_avg[seq_idx].size(-1))))
        # x=np.array(list(range(G_list[seq_idx].size(-1))))

        plt.figure()
        # plt.plot(np.log10(x[1:]), np.log10(abs(G_avg.numpy().mean(axis=0))), linestyle='-', marker='o', label='log|G_avg|', color='blue', lw=2)
        # plt.plot(np.log10(x[1:]), np.log10(abs(G_list[seq_idx].numpy().mean(axis=(0,1)))), linestyle='-', marker='o', label='log|G_avg|', color='blue', lw=2)
        # plt.errorbar(x, G_avg[seq_idx].numpy().mean(axis=0), yerr=G_std[seq_idx].numpy().mean(axis=0)/np.sqrt(750), linestyle='-', marker='o', label='log|G_avg|', color='blue', lw=2)
        plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(650), linestyle='-', marker='o', label='log|G_avg|', color='blue', lw=2)
        plt.xscale('log')
        plt.yscale('log')

        # Add labels and title
        plt.xlabel('X-axis label')
        plt.ylabel('log10(|G|) values')
        plt.title(r"Ntau=20 Nx=20 Ny=20 Nswp=750")
        plt.legend()


        ## Slope annotation
        xs, ys = np.log10(x+1), np.log10(abs(G_avg.numpy().mean(axis=0)))
        mid_idx = len(xs) // 2

        # Get the coordinates of the 0th and midpoint
        x0, y0 = xs[0], ys[0]
        x_mid, y_mid = xs[mid_idx-2], ys[mid_idx-2]
        slope = (y_mid - y0) / (x_mid - x0)

        # Plot the points and the connecting line
        plt.plot(xs, ys, 'bo-', label="Data")
        plt.plot([x0, x_mid], [y0, y_mid], 'r-', linewidth=2, label=f"Line (slope={slope:.2f})")

        # Annotate the slope
        mid_x_pos = (x0 + x_mid) / 2
        mid_y_pos = (y0 + y_mid) / 2
        plt.annotate(f"Slope = {slope:.2f}", (mid_x_pos, mid_y_pos), textcoords="offset points", xytext=(10,10), ha='center', fontsize=12, color='red')

        # save_plot
        class_name = self.__class__.__name__
        method_name = "greens_loglog"
        save_dir = os.path.join(script_path, f"figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")


def load_visualize_final_greens():
    """
    Visualize green functions with error bar
    """
    # Load numerical data
    data_folder_num = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/"

    Lx, Ly, Ltau = 20, 20, 20
    num_site = Lx*Ly*Ltau
    step = 5989464
    swp = math.ceil(step / num_site)
    filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_point/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    res = torch.load(filename)

    G_list = res['G_list']
    G_avg = res['G_avg']
    x = np.array(list(range(G_avg.size(-1))))
    seq_idx = torch.arange(100 * num_site, 750 * num_site, num_site)

    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(650), linestyle='-', marker='o', label='|G_avg|', color='blue', lw=2)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$|G(\tau)|$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={swp}")

    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens"
    save_dir = os.path.join(script_path, f"figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")
    

def load_visualize_final_greens_loglog(Lsize=(20, 20, 20), step=1000001):
    """
    Visualize green functions with error bar
    """
    # Load numerical data

    # Lx, Ly, Ltau = 20, 20, 20
    Lx, Ly, Ltau = Lsize

    num_site = Lx*Ly*Ltau
    # step = 1600000
    swp = math.ceil(step / num_site)
    filename = f"/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/check_points/check_point_local_update/ckpt_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_step_{step}.pt"
    res = torch.load(filename)

    G_list = res['G_list']
    G_avg = res['G_avg']
    x = np.array(list(range(G_avg.size(-1))))
    start, end = 100, 800
    seq_idx = np.arange(start * num_site, end * num_site, 2*num_site)
    # seq_idx = np.arange(start * num_site, end * num_site, int(num_site//10))

    ## Plot
    plt.figure()
    plt.errorbar(x, G_list[seq_idx].numpy().mean(axis=(0, 1)), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='-', marker='o', label='G_avg', color='blue', lw=2)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$|G(\tau)|$")
    plt.title(f"Ntau={Ltau} Nx={Lx} Ny={Ly} Nswp={swp}")

    # -------- Log plot -------- #
    plt.figure()
    plt.errorbar(x+1, abs(G_list[seq_idx].numpy().mean(axis=(0, 1))), yerr=G_list[seq_idx].numpy().std(axis=(0, 1))/np.sqrt(seq_idx.size), linestyle='', marker='o', label='G_avg', color='blue', lw=2)

    G_mean = G_list[seq_idx].numpy().mean(axis=(0, 1))


    # Load and plot analytical data
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


    # save_plot
    class_name = __file__.split('/')[-1].replace('.py', '')
    method_name = "greens_loglog"
    save_dir = os.path.join(script_path, f"figure_{class_name}")
    os.makedirs(save_dir, exist_ok=True) 
    file_path = os.path.join(save_dir, f"{method_name}_{Lx}.pdf")
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved at: {file_path}")


if __name__ == '__main__':

    hmc = LocalU1Sampler()
    # G_avg, G_std = hmc.measure()
    # hmc.total_monitoring()

    # Lx, Ly, Ltau = 20, 20, 20
    # step = 5989464
    Lx, Ly, Ltau = 10, 10, 20
    step = int(16e5)
    load_visualize_final_greens_loglog((Lx, Ly, Ltau), step)
    # load_visualize_final_greens()

    # hmc.step = 10000
    # G_avg, G_std = hmc.resume_measure()

    # hmc.total_monitoring()
    # hmc.visualize_final_greens(G_avg, G_std)
    # hmc.visualize_final_greens_loglog(G_avg, G_std)

    plt.show(block=True)
    dbstop = 1
