import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')

from qed_fermion.coupling_mat import initialize_coupling_mat

class HmcSampler(object):
    def __init__(self, config=None):
        self.Lx = 10
        self.Ly = 12
        self.Ltau = 20
        self.J = 1
        self.boson = None
        self.A = initialize_coupling_mat(self.Lx, self.Ly, self.Ltau, self.J)

        # A = self.A
        # A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
        # A = A.reshape([self.Ltau * self.Ly * self.Lx * 2, self.Ltau * self.Ly * self.Lx * 2])
        # eigenvalues = torch.linalg.eigvalsh(A)  # Optimized for symmetric matrices
        # print(eigenvalues)
        # atol = 1e-5
        # assert  torch.all(eigenvalues >= -atol)

        # Plot
        self.num_tau = self.Ltau
        self.polar = 0  # 0: x, 1: y

        # Statistics
        self.N_therm_step = 30
        self.N_step = 100
        self.step = 0
        self.G_list = torch.zeros(self.N_step, self.num_tau)
        self.accp_list = torch.zeros(self.N_step, dtype=torch.bool)

        # Leapfrog
        self.delta_t = 0.1
        self.N_leapfrog = 50

        # Debug
        
        
    
    def initialize_boson(self):
        """
        Initialize with zero flux across all imaginary time. This amounts to shift of the gauge field and consider only the deviation from the ground state.

        :return: None
        """
        # self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau)
        self.boson = torch.randn(2, self.Lx, self.Ly, self.Ltau) * 0.1
        
    def draw_momentum(self):
        """
        Draw momentum tensor from gaussian distribution.
        :return: [2, Lx, Ly, Ltau] gaussian tensor
        """
        return torch.randn(2, self.Lx, self.Ly, self.Ltau)

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
        H0 = self.action(p, x).item()
        dt = self.delta_t
        
        # # Initialize plot
        # plt.ion()  # Turn on interactive mode
        # fig, ax = plt.subplots()
        # Hs = [H0]

        # # Plot setup
        # line, = ax.plot(Hs, marker='o', linestyle='-', color='b', label='H_s')
        # ax.set_xlabel('Leapfrog Step')
        # ax.set_ylabel('Hamiltonian (H)')
        # ax.set_title('Real-Time Evolution of H_s')
        # ax.legend()
        # plt.grid()
        for i in range(self.N_leapfrog):
            x = x + dt * p
            p = p + dt * self.force(x)
            x = x + dt * p

            # Hs.append(self.action(p, x).item())  # Append new H value

            # # Update plot
            # line.set_ydata(Hs)
            # line.set_xdata(range(len(Hs)))
            # ax.relim()  # Recalculate limits
            # ax.autoscale_view()  # Rescale view

            # plt.draw()
            # plt.pause(0.01)  # Pause for smooth animation

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
        return kinetic + potential

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
        H_old = self.action(p_old, self.boson)
        H_new = self.action(p_new, boson_new)
        print(H_old, H_new)
        if torch.rand(1) < torch.exp(H_old - H_new):
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
        boson_polar = boson[self.polar]
        for dtau in range(self.num_tau):
            if dtau == 0:
                corr = torch.mean(boson_polar * boson_polar, dim=(0, 1, 2))
            else:
                corr = torch.mean(boson_polar[..., :-dtau] * boson_polar[..., dtau:], dim=(0, 1, 2))
            correlations.append(corr)

        return torch.stack(correlations)
    
    def measure(self):
        """
        Do self.N_step metropolis updates, compute greens function for each sample, and store them in self.G_list. Also store the acceptance result in self.accp_list.

        :return: G_avg, G_std
        """
        self.initialize_boson()

        # Thermalize
        for i in range(self.N_therm_step):
            _, accp = self.metropolis_update()

        # Take sample
        for i in tqdm(range(self.N_step)):
            boson, accp = self.metropolis_update()
            self.accp_list[i] = accp
            self.G_list[i] = self.greens_function(boson) if accp else self.G_list[-1]

            # Visualization
            # self.total_monitoring()

            dbstop = 1

        return self.G_list.mean(dim=0), self.G_list.std(dim=0)

    # ------- Visualization -------
    def monitor_accp(self):
        """
        At each metropolis step, monitor the self.G_list (which is a vector, so just monitor its first, middle and rear components) and self.accp_list, and plot them out against steps count. Also monitor the acceptance rate againt the steps.

        :return: None
        """
        plt.plot(self.accp_list.numpy())
        plt.xlabel("Steps")
        plt.ylabel("Acceptance Rate")
        plt.show()

    def visualize_greens_avg_3point(self):
        """
        Visualize the final average greens function which is a vector as a function of dtau.

        :return: None
        """

        idx = [0, self.num_tau // 2, -1]
        plt.plot(self.G_list[:, idx[0]].numpy(), label=f'G[0]')
        plt.plot(self.G_list[:, idx[1]].numpy(), label=f'G[{self.num_tau // 2}]')
        plt.plot(self.G_list[:, idx[2]].numpy(), label=f'G[-1]')
        plt.xlabel("Steps")
        plt.ylabel("Greens Function")
        plt.legend()
        plt.show()

    def total_monitoring(self):
        """
        Visualize obsrv and accp in subplots.
        """
        plt.figure()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(self.accp_list.numpy())
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Acceptance Rate")
        axes[0].set_title("Acceptance Rate Over Steps")

        idx = [0, self.num_tau // 2, -1]
        axes[1].plot(self.G_list[:, idx[0]].numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[:, idx[1]].numpy(), label=f'G[{self.num_tau // 2}]')
        axes[1].plot(self.G_list[:, idx[2]].numpy(), label=f'G[-1]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

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
