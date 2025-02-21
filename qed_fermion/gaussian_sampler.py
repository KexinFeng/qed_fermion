import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')

from qed_fermion.hmc_sampler import HmcSampler
from qed_fermion.coupling_mat2 import initialize_coupling_mat
import numpy as np

script_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = 'cpu'
print(f"device: {device}")

class GaussianSampler(HmcSampler):
    def __init__(self, config=None):
        self.Lx = 10
        self.Ly = 10
        self.Ltau = 20
        self.J = 1.1
        self.K = 1
        self.boson = None
        # [2, Lx, Ly, Ltau,  2, Lx, Ly, Ltau]
        self.A = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, self.J, K=self.K)[0].to(device)
        # self.A[0, 0, 0, 0,  0, 0, 0, 0] += 1 

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 20
        self.N_step = int(1e4)
        self.step = 0
        self.cur_step = self.N_step
        self.thrm_bool = False
        self.G_list = torch.zeros(self.N_step, self.num_tau + 1)
        self.accp_list = torch.zeros(self.N_step, dtype=torch.bool)
        self.accp_rate = torch.zeros(self.N_step)
        self.S_list = torch.zeros(self.N_step + self.N_therm_step)
        # self.H_list = torch.zeros(self.N_step + self.N_therm_step)

        # Debug
        torch.manual_seed(0)
        self.debug_pde = False

        # Numeric
        self.chol = False

    def force_batch(self, x):
        """
        F = -dS/dx = -Ax

        :param x: [bs, 2, Lx, Ly, Ltau] tensor
        :return: evaluation of the force at given x. [bs, 2, Lx, Ly, Ltau]
        """
        dim = 2 * self.Lx * self.Ly * self.Ltau
        A = self.A.reshape(dim, dim)
        tmp =  - A @ x.permute(1, 2, 3, 4, 0).view(dim, -1)
        return tmp.T.view(-1, 2, self.Lx, self.Ly, self.Ltau)
        # return -torch.einsum('ijklmnop,bmnop->bijkl', self.A, x)

    def action_batch(self, boson):
        """
        The action S = 1/2 * boson.T * self.A * boson + momentum**2. The prob ~ e^{-S}.

        :param boson: [bs, 2, Lx, Ly, Ltau] tensor
        :return: the action [bs]
        """
        potential = 1/2 * torch.einsum('bijkl,bijkl->b', boson, -self.force_batch(boson))
        return potential

    def greens_function_batch(self, boson):
        """
        Evaluate the greens function of boson.
        obsrv = phi^{a}_r1 phi^{a}_r2.
        r = (x, y, tau), x1 = x2, y1 = y2, tau2 = tau1 + dtau
        obsrv = [mean(phi[a, :, :, :_taus] * phi[a, :, :, :_taus + dtau], axis = [0, 1, 2]) for dtau in range(0, num_dtau)]

        :param boson: [batch_size, 2, Lx, Ly, Ltau] tensor
        :return: a vector of shape [bs, num_dtau]
        """
        correlations = []
        boson_elem = boson[:, self.polar]
        for dtau in range(self.num_tau + 1):
            idx1 = list(range(self.Ltau))
            idx2 = [(i+dtau) % self.Ltau for i in idx1]
            corr = torch.mean(boson_elem[..., idx1] * boson_elem[..., idx2], dim=(1, 2, 3))
            correlations.append(corr)

        return torch.stack(correlations).T

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

        :return: G_avg, G_std torch.tensor
        """
        dim = self.Ltau * self.Ly * self.Lx * 2
        A = self.A.permute([3, 2, 1, 0, 7, 6, 5, 4]) # [2, Lx, Ly, Ltau]
        A = A.view(dim, dim)

        # # Multivariant Gaussian sampling
        # eigvals, eigvecs = np.linalg.eigh(A.cpu().numpy()) # torch.linalg.eigh
        # eigvals = torch.from_numpy(eigvals).to(device)
        # eigvecs = torch.from_numpy(eigvecs).to(device)
        # # eigvals, eigvecs = torch.linalg.eigh(A.to('cpu'))
        # # eigvals = eigvals.to(device)
        # # eigvecs = eigvecs.to(device)
        
        # # Ensure positive definiteness by clamping small/negative eigenvalues
        # eigvals = torch.clamp(eigvals, min=0)
        
        # # Construct sqrt(A) using modified eigenvalues
        # L = eigvecs @ torch.diag(torch.sqrt(eigvals))
        
        # # Generate standard normal samples
        # z = torch.randn((A.shape[0], self.N_step), device=device)
        # bosons = L @ z  # Transform to get boson samples [dim, bs]

        # Multivariant Gaussian sampling 2
        z = torch.randn((dim, self.N_step), device=device)  # [dim, bs]
        
        if self.chol:
            try:
                eps = 1e-5
                A_reg = A + eps * torch.eye(dim, device=A.device)
                L = torch.linalg.cholesky(A_reg.cpu())  # A = LL^H
                L = L.to(device)
                # S ^ {- 1/2 * boson^T L L^H boson}
                bosons = torch.linalg.solve(L.T, z) # L^H @ boson = z
                # Transform to get boson samples [dim, bs]
            except RuntimeError as e:
                error_info = str(e)
                raise RuntimeError(f"An error occurred: {error_info}")
        else:
            eps = 1e-5
            # A += eps * torch.eye(dim, device=A.device)
            # eigvecs.T @ A @ eigvecs = diag(eigvals)
            # eigvecs @ psi = boson
            eigvals, eigvecs = np.linalg.eigh(A.cpu().numpy())
            eigvals = torch.from_numpy(eigvals).to(device)
            eigvecs = torch.from_numpy(eigvecs).to(device)

            eigvals = torch.clamp(eigvals, min=eps)
            eigvals[0: self.Lx * self.Ly+1] = 1e12

            # L_trans = torch.diag(eigvals**(-1/2)) @ eigvecs.T
            # bosons = L_trans @ z
            plt.semilogy(eigvals.cpu(), '*', linestyle='')

            bosons = eigvecs @ torch.diag(eigvals**(-1/2)) @ z

            # L_trans @ phi = z
            # bosons = torch.linalg.solve(L_trans, z) # [dim, bs]
            # bosons = torch.linalg.solve(A_inv_sqrt, z) # A_inv_sqrt @ z

        # Reshape bosons back to expected shape
        bosons = bosons.T.reshape(self.N_step, 2, self.Lx, self.Ly, self.Ltau)

        # Greens function
        # res = self.greens_function_batch(bosons).cpu()  # res: [bs, num_tau]
        res = self.curl_greens_function_batch(bosons).cpu().T
        self.G_list = res.T
        self.S_list = self.action_batch(bosons).cpu()

        return res.mean(dim=-1), res.std(dim=-1)/torch.tensor([self.N_step])**(1/2)


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

        # save_plot
        class_name = self.__class__.__name__
        method_name = "greens_loglog"
        save_dir = os.path.join(script_path, f"figure_{class_name}")
        os.makedirs(save_dir, exist_ok=True) 
        file_path = os.path.join(save_dir, f"{method_name}_{self.Lx}.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")


if __name__ == '__main__':

    folder = "/Users/kx/Desktop/hmc/qed_fermion/data_gaussian_curl/"
    gmc = GaussianSampler()

    # Execution
    G_avg, G_std = gmc.measure()

    gmc.total_monitoring()
    gmc.visualize_final_greens(G_avg, G_std)
    gmc.visualize_final_greens_loglog(G_avg, G_std)

    # save
    os.makedirs(folder, exist_ok=True)
    file_name = f"./L_{gmc.Ltau}.pt"
    res = {'G_avg': G_avg,
           'G_std': G_std,
           'G_list': gmc.G_list}
    torch.save(res, folder + file_name)

    plt.show()


    # # Loading
    # file_name = f"/L_{22}.pt"
    # loaded = torch.load(folder + file_name)
    # gmc.visualize_final_greens(loaded['G_avg'], loaded['G_std'])
    # gmc.visualize_final_greens_loglog(loaded['G_avg'], loaded['G_std'])
    # plt.show()

    dbstop = 1

