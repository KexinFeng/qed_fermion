import math
import torch
import torch.nn as nn
import time

# Inner loop wrapped in nn.Module
class InnerLoopModule(nn.Module):
    def __init__(self, eta, G_eta, indices_r2, Ltau, Ly, Lx, inner_batch_size):
        super().__init__()
        self.register_buffer('eta', eta)
        self.register_buffer('G_eta', G_eta)
        self.register_buffer('indices_r2', indices_r2)
        self.Ltau, self.Ly, self.Lx = Ltau, Ly, Lx
        self.inner_batch_size = inner_batch_size

    def forward(self, start_idx, end_idx):
        eta_conj = self.eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = self.G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0] // 2
        eta_conj1, G_eta1 = eta_conj[:Nrv], G_eta[:Nrv]
        eta_conj2, G_eta2 = eta_conj[Nrv:], G_eta[Nrv:]

        G_sum = torch.zeros((self.Ly, self.Lx), device=self.eta.device)

        for inner_start in range(start_idx, end_idx, self.inner_batch_size):
            inner_end = min(inner_start + self.inner_batch_size, end_idx)

            sa_batch = self.indices_r2[inner_start:inner_end, 0]
            sb_batch = self.indices_r2[inner_start:inner_end, 1]

            a = eta_conj1[sa_batch] * G_eta1[sb_batch]
            b = G_eta2[sa_batch] * eta_conj2[sb_batch]

            c = torch.fft.ifftn(a * b, dim=(-2, -1))
            result = c.mean(dim=(0, 1)) * ((inner_end - inner_start) / self.indices_r2.shape[0])
            G_sum += result.real

        return G_sum

# Example parameters
Nrv, Ltau, Ly, Lx = 64, 8, 32, 32
total_pairs = math.comb(Nrv, 2)
outer_batch_size = 1024
inner_batch_size = 256

# Example tensors
device = torch.device('cuda')
eta = torch.randn(2*Nrv, Ltau * Ly * Lx, dtype=torch.cfloat, device=device)
G_eta = torch.randn(2*Nrv, Ltau * Ly * Lx, dtype=torch.cfloat, device=device)
indices_r2 = torch.triu_indices(Nrv, Nrv, offset=1).t().to(device)

# Instantiate and compile
inner_model = InnerLoopModule(eta, G_eta, indices_r2, Ltau, Ly, Lx, inner_batch_size).to(device)
compiled_inner_model = torch.compile(inner_model, mode='reduce-overhead', fullgraph=True)

# Warm-up
compiled_inner_model(0, outer_batch_size)

# Actual nested loop execution
G_sum = torch.zeros((Ly, Lx), device=device)
num_batches = (total_pairs + outer_batch_size - 1) // outer_batch_size

# Timing the execution
torch.cuda.synchronize()
start = time.time()

for batch_idx in range(num_batches):
    start_idx = batch_idx * outer_batch_size
    end_idx = min(start_idx + outer_batch_size, total_pairs)
    G_sum += compiled_inner_model(start_idx, end_idx)

torch.cuda.synchronize()
end = time.time()

print(f"Nested loops with torch.compile executed in {end - start:.4f}s")
