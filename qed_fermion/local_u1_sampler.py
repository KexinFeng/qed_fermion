import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')

from qed_fermion.coupling_mat2 import initialize_coupling_mat
from qed_fermion.hmc_sampler import HmcSampler

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

class LocalU1Sampler(HmcSampler):
    def __init__(self, N_step=10000, config=None):
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
        self.N_therm_step = 500
        self.N_step = N_step
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
        self.w = 0.001 * torch.pi

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False
        self.debug = True

    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau)
        self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau, device=device) * 0.1

    def force(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x.
        """
        return -torch.einsum('ijklmnop,mnop->ijkl', self.A, x)

    def action(self, boson):
        """
        The action S = 1/2 * boson.transpose * self.A * boson.

        :param boson: [2, Lx, Ly, Ltau] tensor
        :return: the action
        """
        potential = 0.5 * torch.einsum('ijkl,ijkl->', boson, -self.force(boson))
        return potential
    
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

        # Select tau index based on the current step
        idx = self.step % self.Ltau

        # Sample new values in the window [boson - w, boson + w], with periodicity mod 2π
        delta = (torch.rand_like(boson_new[:, :, :, idx]) - 0.5) * 2 * self.w  # Uniform in [-w, w]
        boson_new[:, :, :, idx] += delta

        # Apply periodic boundary condition: ensure values remain in [-π, π] mod 2π
        boson_new[:, :, :, idx] = (boson_new[:, :, :, idx] + torch.pi) % (2 * torch.pi) - torch.pi

        return boson_new
    
    
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
        boson_new = self.local_u1_proposer()
        S_old = self.action(self.boson)
        S_new = self.action(boson_new)

        accp =  torch.rand(1, device=device) < torch.exp(S_old - S_new)

        if self.debug and self.thrm_bool:
            print(f"S_old, S_new, diff: {S_old}, {S_new}, {S_old - S_new}")
            print(f"threshold: {torch.exp(S_old - S_new).item()}")
            print(accp.item())
        if accp:
            self.boson = boson_new
            return self.boson, True, S_new
        else:
            return self.boson, False, S_old

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

        # Thermalize
        self.thrm_bool = True
        for i in range(self.N_therm_step):
            boson, accp, S = self.metropolis_update()
            self.S_list[i] = S
            self.step += 1

        self.G_list[-1] = self.greens_function(boson)

        # Take sample
        self.thrm_bool = False
        for i in tqdm(range(self.N_step)):
            boson, accp, S = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float)).item()
            self.G_list[i] = self.greens_function(boson) if accp else self.G_list[i-1]
            self.S_list[i + self.N_therm_step] = S
            self.step += 1

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
            boson, accp, S = self.metropolis_update()
            self.accp_list[i] = accp
            self.accp_rate[i] = torch.mean(self.accp_list[:i+1].to(torch.float)).item()
            self.G_list[i] = self.greens_function(boson) if accp else self.G_list[i-1]
            self.S_list[i + self.N_therm_step] = S
            self.step += 1

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


if __name__ == '__main__':

    hmc = LocalU1Sampler(N_step=10000)
    # hmc.N_step = 50000
    # G_avg, G_std = hmc.measure()

    # hmc.total_monitoring()
    # hmc.visualize_final_greens(G_avg)

    hmc.step = 201500
    G_avg, G_std = hmc.resume_measure()

    # print(hmc.accp_list)

    # plt.figure()
    # # hmc.visualize_final_greens(G_avg)
    # plt.plot(hmc.accp_list.numpy())
    # plt.xlabel("Steps")
    # plt.ylabel("Acceptance Rate")
    # plt.show()

    hmc.total_monitoring()
    hmc.visualize_final_greens(G_avg)

    plt.show(block=True)
    dbstop = 1
