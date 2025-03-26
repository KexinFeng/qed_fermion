import torch
import sys, os
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.hmc_sampler_batch import HmcSampler
import os

class CgHmcSampler(HmcSampler):
    def __init__(self, J=0.5, Nstep=3000, config=None):
        super().__init__(J, Nstep, config)

    def preconditioned_cg(self, M, b, tol=1e-8, max_iter=100):
        """
        Solve M'M x = b using preconditioned conjugate gradient (CG) algorithm.

        :param M: Sparse matrix M from get_M_sparse()
        :param b: Right-hand side vector
        :param tol: Tolerance for convergence
        :param max_iter: Maximum number of iterations
        :return: Solution vector x
        """
        # Initialize variables
        x = torch.zeros_like(b, dtype=torch.complex64, device=b.device)
        r = b.clone()
        z = r.clone()  # Preconditioner is identity for now
        p = z.clone()
        rz_old = torch.dot(r.conj(), z).real

        errors = []

        for i in range(max_iter):
            # Matrix-vector product with M'M
            Mp = torch.sparse.mm(M, p)
            Mtp = torch.sparse.mm(M.T.conj(), Mp)
            
            alpha = rz_old / torch.dot(p.conj(), Mtp).real
            x += alpha * p
            r -= alpha * Mtp

            # Compute and store the error (norm of the residual)
            error = torch.norm(r).item()
            errors.append(error)

            # Check for convergence
            if error < tol:
                print(f"Converged in {i+1} iterations.")
                break

            z = r.clone()  # Preconditioner is identity for now
            rz_new = torch.dot(r.conj(), z).real
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        # Plot the error and save the figure
        plt.figure()
        plt.plot(errors, label="Residual Norm")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Error (Log Scale)")
        plt.title("Convergence of Preconditioned CG")
        plt.legend()

        # Save the plot
        script_path = os.path.dirname(os.path.abspath(__file__))
        class_name = __file__.split('/')[-1].replace('.py', '')
        method_name = "preconditioned_cg"
        save_dir = os.path.join(script_path, f"./figures/{class_name}")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{method_name}_convergence.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")

        return x

    def benchmark(self, bs=5, device='cpu'):
        """
        Benchmark the preconditioned CG solver.

        :param bs: Batch size
        :param device: Device to run the benchmark on ('cpu' or 'cuda')
        """
        # Fix the random seed for reproducibility
        torch.manual_seed(25)

        # Generate random boson field with uniform randomness across all dimensions
        self.boson = (torch.rand(bs - 1, 2, self.Lx, self.Ly, self.Ltau, device=device) - 0.5) * 0.2

        curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        delta_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
        self.boson = torch.cat([self.boson, delta_boson], dim=0)

        # Initialize right-hand side vector b
        b = self.draw_psudo_fermion()

        # Benchmark preconditioned CG for each boson in the batch
        convergence_steps = []
        for i in range(bs):
            boson = self.boson[i]
            M, _ = self.get_M_sparse(boson)
            x = self.preconditioned_cg(M, b, tol=1e-8, max_iter=1000)
            convergence_steps.append(x)

        return sum(convergence_steps) / len(convergence_steps)

if __name__ == '__main__':
    sampler = CgHmcSampler(J=0.5, Nstep=3000, config=None)
    sampler.Lx, sampler.Ly, sampler.Ltau = 6, 6, 10
    results = sampler.benchmark(bs=10, device='cpu')
    print("Benchmark results:", results)
