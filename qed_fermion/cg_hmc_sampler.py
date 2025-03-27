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
import pandas as pd

import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

import matlab.engine


class CgHmcSampler(HmcSampler):
    def __init__(self, J=0.5, Nstep=3000, config=None):
        super().__init__(J, Nstep, config)
        self.plt_cg = True

    def preconditioned_cg(self, MhM, b, MhM_inv=None, tol=1e-8, max_iter=100, b_idx=None):
        """
        Solve M'M x = b using preconditioned conjugate gradient (CG) algorithm.

        :param M: Sparse matrix M from get_M_sparse()
        :param b: Right-hand side vector
        :param tol: Tolerance for convergence
        :param max_iter: Maximum number of iterations
        :return: Solution vector x
        """
        # Initialize variables
        x = torch.zeros_like(b).view(-1, 1)
        r = b.view(-1, 1) - torch.sparse.mm(MhM, x)
        z = torch.sparse.mm(MhM_inv, r) if MhM_inv else r
        p = z
        rz_old = torch.dot(r.conj().view(-1), z.view(-1)).real

        errors = []

        fig, axs = plt.subplots(1, figsize=(12, 7.5))  # Two rows, one column

        residuals = []
        for i in range(max_iter):
            # Matrix-vector product with M'M
            Op = torch.sparse.mm(MhM, p)
            
            alpha = rz_old / torch.dot(p.conj().view(-1), Op.view(-1)).real
            x += alpha * p
            r -= alpha * Op

            # Compute and store the error (norm of the residual)
            error = torch.norm(r).item() / torch.norm(b).item()
            errors.append(error)
            residuals.append(error)

            if self.plt_cg:
                # Plot intermediate results
                axs.cla()
                axs.plot(residuals, marker='o', linestyle='-', color='b', label='Residual Norm')
                axs.set_ylabel('Residual Norm')
                axs.set_yscale('log')
                axs.legend()
                axs.grid()
                axs.set_title(f'{self.Lx}x{self.Ly}x{self.Ltau} Lattice b_idx={b_idx}')

                plt.pause(0.01)  # Pause to update the plot

            # Check for convergence
            if error < tol:
                print(f"Converged in {i+1} iterations.")
                break

            z = torch.sparse.mm(MhM_inv, r) if MhM_inv else r  # Apply preconditioner to r
            rz_new = torch.dot(r.conj().view(-1), z.view(-1)).real
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        # Save the plot
        script_path = os.path.dirname(os.path.abspath(__file__))
        class_name = __file__.split('/')[-1].replace('.py', '')
        method_name = "preconditioned_cg"
        save_dir = os.path.join(script_path, f"./figures/{class_name}")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{method_name}_convergence.pdf")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved at: {file_path}")

        return x, errors[-1]
    
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
                                # ncv=50, maxiter=300000)[0][0]
        # sigma_min = splinalg.eigsh(A_scipy, k=1, which='SM', tol=tol)[0][0]
                                # ncv=50, maxiter=300000)[0][0]                                
        try:
            sigma_min = splinalg.eigsh(A_scipy, k=1, which='SM', tol=tol)[0][0]
                                # ncv=50, maxiter=300000)[0][0]
        except splinalg.ArpackNoConvergence as e:
            print(f"Warning: {e}")
            if len(e.eigenvalues) > 0:
                sigma_min = e.eigenvalues[0]  # This is the smallest eigenvalue among the converged ones
                print(f"Using the smallest available eigenvalue: {sigma_min}")
            else:
                sigma_min = 1e-12  # Fallback to a very small value
                print("No eigenvalues available. Falling back to a default value of 1e-12.")

        return sigma_max / max(sigma_min, 1e-12), sigma_max, max(sigma_min, 1e-12)


    def get_precon(self, pi_flux_boson):
        MhM, _, _ = self.get_M_sparse(pi_flux_boson)
        retrieved_indices = MhM.indices() + 1  # Convert to 1-based indexing for MATLAB
        retrieved_values = MhM.values()

        # Pass indices and values to MATLAB
        matlab_function_path = '/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/utils/'
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_function_path)

        # Convert indices and values directly to MATLAB format
        matlab_indices = matlab.double(retrieved_indices.cpu().tolist())
        matlab_values = matlab.double(retrieved_values.cpu().tolist(), is_complex=True)

        # Call MATLAB function
        result_indices, result_values = eng.preconditioner(
            matlab_indices, matlab_values, MhM.size(0), MhM.size(1), nargout=2
        )
        eng.quit()

        # Convert MATLAB results directly to PyTorch tensors
        result_indices = torch.tensor(result_indices, dtype=torch.long, device=MhM.device) - 1
        result_values = torch.tensor(result_values, dtype=MhM.dtype, device=MhM.device)

        # Create MhM_inv as a sparse_coo_tensor
        MhM_inv = torch.sparse_coo_tensor(
            result_indices,
            result_values,
            size=MhM.shape
        ).coalesce()

        return MhM_inv

    def benchmark(self, bs=5):
        """
        Benchmark the preconditioned CG solver.

        :param bs: Batch size
        :param device: Device to run the benchmark on ('cpu' or 'cuda')
        """
        # Fix the random seed for reproducibility
        torch.manual_seed(25)

        # Generate random boson field with uniform randomness across all dimensions
        self.boson = (torch.rand(bs, 2, self.Lx, self.Ly, self.Ltau, device=self.boson.device) - 0.5) * torch.linspace(0.5 * 3.14, 2 * 3.14, bs, device=self.boson.device).reshape(-1, 1, 1, 1, 1)

        # Pi flux
        curl_mat = self.curl_mat * torch.pi / 4  # [Ly*Lx, Ly*Lx*2]
        boson = curl_mat[self.i_list_1, :].sum(dim=0)  # [Ly*Lx*2]
        boson = boson.repeat(1 * self.Ltau, 1)
        pi_flux_boson = boson.reshape(1, self.Ltau, self.Ly, self.Lx, 2).permute([0, 4, 3, 2, 1])
        self.boson = torch.cat([self.boson, pi_flux_boson], dim=0)

        # Generate new bosons centered around pi_flux_boson with varying sigma
        sigmas = torch.linspace(0.1 * 3.14/2, 1*3.14/2, bs-1, device=self.boson.device)
        new_bosons = torch.stack([
            pi_flux_boson.squeeze(0) + torch.randn_like(pi_flux_boson.squeeze(0)) * sigma
            for sigma in sigmas
        ])
        self.boson = torch.cat([self.boson, new_bosons], dim=0)

        # Initialize right-hand side vector b
        b = self.draw_psudo_fermion().squeeze(0)

        # Get preconditioner
        precon = self.get_precon(pi_flux_boson)

        # Benchmark preconditioned CG for each boson in the batch
        convergence_steps = []
        condition_numbers = []
        sig_max_values = []
        sig_min_values = []
        blk_sparsities = []
        residual_errors = []
        for i in range(bs*2):
            if not i in [1]: 
                continue

            boson = self.boson[i]
            MhM, _, blk_sparsity = self.get_M_sparse(boson)
            # Check if M'M is correct
            if self.Ltau < 20:
                M, _ = self.get_M(boson.unsqueeze(0))
                torch.testing.assert_close(M.T.conj() @ M, MhM.to_dense(), rtol=1e-5, atol=1e-5)

            # # Compute condition number            
            # cond_number_approx, sig_max, sig_min = self.estimate_condition_number(MhM, tol=1e-3)
            # print(f"Approximate condition number of M'@M: {cond_number_approx}")
            # print(f"Sigma max: {sig_max}, Sigma min: {sig_min}")

            # condition_numbers.append(float(cond_number_approx))
            # sig_max_values.append(float(sig_max))
            # sig_min_values.append(float(sig_min))
            blk_sparsities.append(blk_sparsity)

            # Test cg and preconditioned_cg
            conv_step, r_err = self.preconditioned_cg(MhM, b, tol=1e-3, max_iter=1000, b_idx=i, MhM_inv=precon)
            convergence_steps.append(conv_step)
            residual_errors.append(r_err)

        mean_conv_steps = sum(convergence_steps) / len(convergence_steps) if convergence_steps else None
        mean_condition_num = sum(condition_numbers) / len(condition_numbers) if condition_numbers else None
        mean_sparsity = sum(blk_sparsities) / len(blk_sparsities) if blk_sparsities else None
        mean_sig_max = sum(sig_max_values) / len(sig_max_values) if sig_max_values else None
        mean_sig_min = sum(sig_min_values) / len(sig_min_values) if sig_min_values else None

        return mean_conv_steps, mean_sparsity, mean_condition_num, condition_numbers, mean_sig_max, mean_sig_min, sig_max_values, sig_min_values, residual_errors


if __name__ == '__main__':
    sampler = CgHmcSampler(J=0.5, Nstep=3000, config=None)
    sampler.Lx, sampler.Ly = 10, 10    # initial test: Lx=Ly=6, Ltau=10
    cg_bs = 2

    Ltau_values = [10, 20, 50, 100, 200, 400, 600]
    Ltau_values = [200, 400, 600]
    mean_conv_steps = []
    mean_condition_nums = []
    mean_sparsities = []

    for Ltau in Ltau_values:
        sampler.Ltau = Ltau
        sampler.reset()
        print(f"Start Ltau={sampler.Ltau}... ")

        start_time = time.time()
        results = sampler.benchmark(bs=cg_bs)
        end_time = time.time()
        execution_time = end_time - start_time

        print("Mean convergence steps:", results[0])
        print("Mean sparsities:", results[1])
        print("Mean condition numbers:", results[2])
        print("Condition numbers:", results[3])
        print("Mean sigma max:", results[4])
        print("Mean sigma min:", results[5])
        print("Sigma max values:", results[6])
        print("Sigma min values:", results[7])
        print("Relative residual errors:", results[8])
        print(f"Execution time for Ltau={sampler.Ltau}: {execution_time:.2f} seconds")

        # Convert results to a dictionary
        results_dict = {
            "ltau": sampler.Ltau,
            "mean_conv_steps": results[0],
            "mean_sparsity": results[1],
            "mean_cond_num": results[2],
            "cond_nums": results[3],
            "mean_sig_max": results[4],
            "mean_sig_min": results[5],
            "sig_max_vals": results[6],
            "sig_min_vals": results[7],
            "rel_res_err": results[8],
            "exec_time_s": execution_time
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
        df = pd.DataFrame([results_dict])
        csv_file_path = torch_file_path.replace('.pt', '.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"-----> save to {csv_file_path}")

        # Append
        mean_conv_steps.append(results[0])
        mean_condition_nums.append(results[1])
        mean_sparsities.append(results[2])

        print('\n')

    print("Ltau values:", Ltau_values)
    print("Mean convergence steps:", mean_conv_steps)
    print("Mean condition numbers:", mean_condition_nums)
    print("Mean sparsities:", mean_sparsities)
