import numpy as np
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
import time

import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

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
    
    # Approximate the condition number

    @staticmethod
    def estimate_condition_number(A, tol=1e-6):
        """Estimate the condition number."""
        # Convert torch sparse_coo_tensor to scipy csr_matrix
        A_scipy = sp.csr_matrix((A.values().cpu().numpy(), 
                                (A.indices()[0].cpu().numpy(), 
                                A.indices()[1].cpu().numpy())), 
                                shape=A.shape)

        # Compute the smallest and largest eigenvalues
        sigma_max = splinalg.eigsh(A_scipy, k=1, which='LM', tol=tol)[0][0]
        sigma_min = splinalg.eigsh(A_scipy, k=1, which='SM', tol=tol)[0][0]
        return sigma_max / max(sigma_min, 1e-12), sigma_max, max(sigma_min, 1e-12)
    
    # @staticmethod
    # def power_iteration(A, max_iters=100):
    #     """Estimate the largest singular value using power iteration."""
    #     v = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
    #     v /= torch.norm(v)
    #     for _ in range(max_iters):
    #         v = torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)
    #         v /= torch.norm(v)
    #     return torch.norm(torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1))

    # @staticmethod
    # def inverse_power_iteration(A, max_iters=100, cg_iters=10, tol=1e-6):
    #     """Estimate the smallest singular value using inverse iteration with CG."""
    #     # Convert torch sparse_coo_tensor to scipy csr_matrix
    #     A_scipy = sp.csr_matrix((A.values().cpu().numpy(), 
    #                             (A.indices()[0].cpu().numpy(), 
    #                             A.indices()[1].cpu().numpy())), 
    #                             shape=A.shape)
        
    #     v = torch.randn(A.shape[1], device=A.device, dtype=A.dtype).cpu().numpy()
    #     v /= np.linalg.norm(v)
    #     for _ in range(max_iters):
    #         # Use conjugate gradient to solve (A^T A)w = v
    #         w, _ = splinalg.cg(A_scipy.T @ A_scipy, v, tol=tol, maxiter=cg_iters)
    #         v = w / np.linalg.norm(w)
    #     return 1.0 / np.linalg.norm(A_scipy @ v)

    # @staticmethod
    # def estimate_condition_number2(A, max_iters=100, cg_iters=10, tol=1e-6):
    #     """Estimate the condition number of a matrix."""
    #     σ_max = CgHmcSampler.power_iteration(A, max_iters)
    #     σ_min = CgHmcSampler.inverse_power_iteration(A, max_iters, cg_iters, tol)
    #     return σ_max / max(σ_min, 1e-12), σ_max, max(σ_min, 1e-12)  # Prevent division by zero

    def benchmark(self, bs=5):
        """
        Benchmark the preconditioned CG solver.

        :param bs: Batch size
        :param device: Device to run the benchmark on ('cpu' or 'cuda')
        """
        # Fix the random seed for reproducibility
        torch.manual_seed(25)

        # Generate random boson field with uniform randomness across all dimensions
        self.boson = (torch.rand(bs, 2, self.Lx, self.Ly, self.Ltau, device=self.boson.device) - 0.5) * 2*3.14

        # Pi flux
        curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        delta_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
        self.boson = torch.cat([self.boson, delta_boson], dim=0)

        # Generate new bosons centered around delta_boson with varying sigma
        sigmas = torch.linspace(0.1 * 3.14/2, 1*3.14/2, bs-1, device=self.boson.device)
        new_bosons = torch.stack([
            delta_boson.squeeze(0) + torch.randn_like(delta_boson.squeeze(0)) * sigma
            for sigma in sigmas
        ])
        self.boson = torch.cat([self.boson, new_bosons], dim=0)

        # Initialize right-hand side vector b
        b = self.draw_psudo_fermion()

        # Benchmark preconditioned CG for each boson in the batch
        convergence_steps = []
        condition_numbers = []
        blk_sparsities = []
        for i in range(bs*2):
            boson = self.boson[i]
            MhM, _, blk_sparsity = self.get_M_sparse(boson)
            # Check if M'M is correct
            if self.Ltau < 50:
                M, _ = self.get_M(boson.unsqueeze(0))
                torch.testing.assert_close(M.T.conj() @ M, MhM.to_dense(), rtol=1e-5, atol=1e-5)

            # Compute condition number            
            cond_number_approx, sig_max, sig_min = self.estimate_condition_number(MhM, tol=1e-6)
            print(f"Approximate condition number of M'@M: {cond_number_approx}")

            # print(f"Sigma max: {sig_max}, Sigma min: {sig_min}")
            # O_dense = MhM.to_dense()

            # a = torch.svd(O_dense)
            # print(f"Smallest singular value (s[-1]): {a.S[-1]}, Largest singular value (s[0]): {a.S[0]}")

            # condition_number = torch.linalg.cond(O_dense)
            # torch.testing.assert_close(condition_number, torch.tensor(cond_number_approx, dtype=condition_number.dtype, device=condition_number.device), rtol=1e-3, atol=1e-5),
            # print(f"Approximate condition number of M'@M: {condition_number}")

            condition_numbers.append(cond_number_approx)
            blk_sparsities.append(blk_sparsity)

            # x = self.preconditioned_cg(MhM, b, tol=1e-8, max_iter=1000)
            # convergence_steps.append(x)

        mean_conv_steps = sum(convergence_steps) / len(convergence_steps) if convergence_steps else None
        mean_condition_num = sum(condition_numbers) / len(condition_numbers) if condition_numbers else None
        mean_sparsity = sum(blk_sparsities) / len(blk_sparsities) if blk_sparsities else None

        return mean_conv_steps, mean_condition_num, mean_sparsity, condition_numbers


if __name__ == '__main__':
    sampler = CgHmcSampler(J=0.5, Nstep=3000, config=None)
    sampler.Lx, sampler.Ly = 10, 10    # initial test: Lx=Ly=6, Ltau=10

    Ltau_values = [10, 20, 50, 100, 200, 400]
    mean_conv_steps = []
    mean_condition_nums = []
    mean_sparsities = []

    for Ltau in Ltau_values:
        sampler.Ltau = Ltau
        sampler.reset()
        print(f"Start Ltau={sampler.Ltau}... ")

        start_time = time.time()
        results = sampler.benchmark(bs=3)
        end_time = time.time()
        execution_time = end_time - start_time

        print("Mean convergence steps:", results[0])
        print("Mean condition numbers:", results[1])
        print("Condition numbers:", results[3])
        print("Mean sparsities:", results[2])
        print(f"Execution time for Ltau={sampler.Ltau}: {execution_time:.2f} seconds\n\n")

        # Convert results to a dictionary
        results_dict = {
            "Ltau": sampler.Ltau,
            "Mean Convergence Steps": results[0],
            "Mean Condition Numbers": results[1],
            "Condition Numbers": results[3],
            "Mean Sparsities": results[2],
            "Execution Time (s)": execution_time
        }

        # Save the results to a .py file
        script_path = os.path.dirname(os.path.abspath(__file__))
        class_name = __file__.split('/')[-1].replace('.py', '')
        save_dir = os.path.join(script_path, f"./results/{class_name}")
        os.makedirs(save_dir, exist_ok=True)

        torch_file_path = os.path.join(save_dir, f"Ltau_{sampler.Ltau}_results.pt")
        torch.save(results_dict, torch_file_path)
        print(f"Results saved to: {torch_file_path}")

        # Save the results to a .csv file
        csv_file_path = os.path.join(save_dir, f"Ltau_{sampler.Ltau}_results.csv")
        with open(csv_file_path, "w") as csv_file:
            csv_file.write("Ltau,Mean Convergence Steps,Mean Condition Numbers,Condition Numbers,Mean Sparsities,Execution Time (s)\n")
            csv_file.write(f"{results_dict['Ltau']},{results_dict['Mean Convergence Steps']},{results_dict['Mean Condition Numbers']},\"{results_dict['Condition Numbers']}\",{results_dict['Mean Sparsities']},{results_dict['Execution Time (s)']}\n")
        print(f"Results saved to: {csv_file_path}")

        # Append
        mean_conv_steps.append(results[0])
        mean_condition_nums.append(results[1])
        mean_sparsities.append(results[2])

    print("Ltau values:", Ltau_values)
    print("Mean convergence steps:", mean_conv_steps)
    print("Mean condition numbers:", mean_condition_nums)
    print("Mean sparsities:", mean_sparsities)
