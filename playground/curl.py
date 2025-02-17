import torch
import time

class Benchmark:
    def __init__(self, Lx, Ly, Ltau, batch_size, num_tau):
        self.Lx = Lx
        self.Ly = Ly
        self.Ltau = Ltau
        self.batch_size = batch_size
        self.num_tau = num_tau

    def curl_greens_function_batch(self, boson):
        batch_size, _, Lx, Ly, Ltau = boson.shape

        phi_x = boson[:, 0]
        phi_y = boson[:, 1]

        curl_phi = (
            phi_x +
            torch.roll(phi_y, shifts=-1, dims=1) - 
            torch.roll(phi_x, shifts=-1, dims=2) -  
            phi_y
        )

        correlations = []
        for dtau in range(self.num_tau + 1):
            idx1 = list(range(Ltau))
            idx2 = [(i + dtau) % Ltau for i in idx1]
            
            corr = torch.mean(curl_phi[..., idx1] * curl_phi[..., idx2], dim=(1, 2))
            correlations.append(corr)

        return torch.stack(correlations)

    def run_benchmark(self, num_runs=10):
        boson = torch.randn(self.batch_size, 2, self.Lx, self.Ly, self.Ltau)

        # Warm-up
        self.curl_greens_function_batch(boson)

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.curl_greens_function_batch(boson)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        print(f"Benchmark for (batch={self.batch_size}, Lx={self.Lx}, Ly={self.Ly}, Ltau={self.Ltau}): {avg_time:.6f} s per run")

# Example runs
benchmark = Benchmark(Lx=16, Ly=16, Ltau=32, batch_size=10, num_tau=32)
benchmark.run_benchmark()

benchmark = Benchmark(Lx=32, Ly=32, Ltau=64, batch_size=10, num_tau=64)
benchmark.run_benchmark()
