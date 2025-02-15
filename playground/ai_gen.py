import matplotlib.pyplot as plt
import torch

class HmcSampler(object):
    def __init__(self, config=None):
        self.Lx = 2
        self.Ly = 3
        self.Ltau = 3
        self.boson = None
        self.A = torch.randn(2, self.Lx, self.Ly, self.Ltau, 2, self.Lx, self.Ly, self.Ltau)

        self.num_dtau = 30
        self.polar = 0  # 0: x, 1: y

        self.N_step = 100
        self.step = 0
        self.G_list = torch.zeros(self.N_step, self.num_dtau)
        self.acp_list = torch.zeros(self.N_step)

    def initialize_boson(self, a):
        self.boson = torch.zeros(2, self.Lx, self.Ly, self.Ltau)

    def draw_momentum(self):
        return torch.randn(2, self.Lx, self.Ly, self.Ltau)

    def force(self, x):
        return -torch.einsum('ijklmnop,mnop->ijkl', self.A, x)

    def leapfrog_proposer(self):
        p = self.draw_momentum()
        x = self.boson.clone()
        dt = 0.1
        for _ in range(10):
            x = x + dt * p
            p = p + dt * self.force(x)
            x = x + dt * p
        return x

    def action(self, momentum, boson):
        kinetic = torch.sum(momentum ** 2) / 2
        potential = 0.5 * torch.einsum('ijkl,ijkl->', boson, self.force(boson))
        return kinetic + potential

    def metropolis_update(self):
        p = self.draw_momentum()
        boson_new = self.leapfrog_proposer()
        H_old = self.action(p, self.boson)
        H_new = self.action(p, boson_new)
        if torch.rand(1) < torch.exp(H_old - H_new):
            self.boson = boson_new

    def greens_function_avg(self):
        for i in range(self.N_step):
            self.metropolis_update()
            self.G_list[i] = self.greens_function(self.boson)
            self.acp_list[i] = 1  # Track acceptance rate
        return self.G_list.mean(dim=0), self.G_list.std(dim=0)

    def monitor(self):
        plt.plot(self.acp_list.numpy())
        plt.xlabel("Steps")
        plt.ylabel("Acceptance Rate")
        plt.show()

    def monitor_greens(self):
        idx = [0, self.num_dtau // 2, -1]
        plt.plot(self.G_list[:, idx[0]].numpy(), label=f'G[0]')
        plt.plot(self.G_list[:, idx[1]].numpy(), label=f'G[{self.num_dtau // 2}]')
        plt.plot(self.G_list[:, idx[2]].numpy(), label=f'G[-1]')
        plt.xlabel("Steps")
        plt.ylabel("Greens Function")
        plt.legend()
        plt.show()

    def total_monitoring(self):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(self.acp_list.numpy())
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Acceptance Rate")
        axes[0].set_title("Acceptance Rate Over Steps")

        idx = [0, self.num_dtau // 2, -1]
        axes[1].plot(self.G_list[:, idx[0]].numpy(), label=f'G[0]')
        axes[1].plot(self.G_list[:, idx[1]].numpy(), label=f'G[{self.num_dtau // 2}]')
        axes[1].plot(self.G_list[:, idx[2]].numpy(), label=f'G[-1]')
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Greens Function")
        axes[1].set_title("Greens Function Over Steps")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def visualize_final_greens(self):
        G_avg, _ = self.greens_function_avg()
        plt.plot(G_avg.numpy())
        plt.xlabel("dtau")
        plt.ylabel("Greens Function")
        plt.show()
