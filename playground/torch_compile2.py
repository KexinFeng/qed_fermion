import torch
import torch.nn as nn
import time

torch.manual_seed(0)

# Inner loop wrapped in nn.Module
class InnerLoopModule(nn.Module):
    def __init__(self, indices_r2, Ltau, Ly, Lx, inner_batch_size):
        super().__init__()
        self.register_buffer('indices_r2', indices_r2)
        self.Ltau, self.Ly, self.Lx = Ltau, Ly, Lx
        self.inner_batch_size = inner_batch_size

    def forward(self, eta, G_eta, start_idx, end_idx):
        eta_conj = eta.conj().view(-1, self.Ltau, self.Ly, self.Lx)
        G_eta = G_eta.view(-1, self.Ltau, self.Ly, self.Lx)

        Nrv = eta_conj.shape[0] // 2
        eta_conj1, G_eta1 = eta_conj[:Nrv], G_eta[:Nrv]
        eta_conj2, G_eta2 = eta_conj[Nrv:], G_eta[Nrv:]

        sa, sb = self.indices_r2[:, 0], self.indices_r2[:, 1]
        num_samples = sa.shape[0]

        G_delta_0_G_delta_0_mean = torch.zeros((self.Ly, self.Lx), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean1 = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)
        G_res_mean2 = torch.zeros((1,), dtype=eta_conj.dtype, device=eta.device)

        for inner_start in range(start_idx, end_idx, self.inner_batch_size):
            inner_end = min(inner_start + self.inner_batch_size, end_idx)

            sa_batch = sa[inner_start:inner_end]
            sb_batch = sb[inner_start:inner_end]

            a = torch.roll(eta_conj1[sa_batch],  shifts=0, dims=-1) * \
                torch.roll(G_eta1[sb_batch],     shifts=0, dims=-1) * \
                torch.roll(eta_conj2[sa_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta2[sb_batch],     shifts=-1, dims=-1)

            b = torch.roll(G_eta1[sa_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj1[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta2[sa_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj2[sb_batch],  shifts=-0, dims=-1)

            c1 = torch.roll(G_eta1[sb_batch],    shifts=-1, dims=-1) *\
                torch.roll(eta_conj1[sb_batch],  shifts=-1, dims=-1) * \
                torch.roll(G_eta2[sa_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj2[sa_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta2[sb_batch],     shifts=-2, dims=-1) * \
                torch.roll(eta_conj2[sb_batch],  shifts=-0, dims=-1)

            c2 = torch.roll(G_eta1[sa_batch],    shifts=-2, dims=-1) *\
                torch.roll(eta_conj1[sa_batch],  shifts=-0, dims=-1) * \
                torch.roll(G_eta1[sb_batch],     shifts=-0, dims=-1) * \
                torch.roll(eta_conj1[sb_batch],  shifts=-2, dims=-1) * \
                torch.roll(G_eta2[sb_batch],     shifts=-1, dims=-1) * \
                torch.roll(eta_conj2[sb_batch],  shifts=-1, dims=-1)

            # FFT
            a_F_neg_k = torch.fft.ifftn(a, (self.Ly, self.Lx), norm="backward")
            b_F = torch.fft.fftn(b, (self.Ly, self.Lx), norm="forward")
            batch_result = torch.fft.ifftn(a_F_neg_k * b_F, (self.Ly, self.Lx), norm="forward")

            factor = (inner_end - inner_start) / num_samples
            G_delta_0_G_delta_0_mean += batch_result.mean(dim=(0, 1)) * factor
            G_res_mean1 += c1.mean(dim=(0, 1, 2, 3)) * factor
            G_res_mean2 += c2.mean(dim=(0, 1, 2, 3)) * factor

        G_delta_0_G_delta_0_mean[0, -1] -= G_res_mean1[0]
        G_delta_0_G_delta_0_mean[0, 1] -= G_res_mean2[0]
        return G_delta_0_G_delta_0_mean

@torch.inference_mode()
def test():
    # Device setup
    device = torch.device("cuda")

    # Parameters for InnerLoopModule
    Ltau, Ly, Lx = 32, 16, 16
    inner_batch_size = 1000
    
    # Create indices_r2 (sample indices)
    num_samples = 10000
    indices_r2 = torch.randint(0, num_samples//2, (num_samples, 2), device=device)
    
    # Model setup
    model = InnerLoopModule(indices_r2, Ltau, Ly, Lx, inner_batch_size).to(device)
    
    # Create input tensors
    batch_size = num_samples
    eta = torch.randn(batch_size, Ltau * Ly * Lx, device=device, dtype=torch.complex64)
    G_eta = torch.randn(batch_size, Ltau * Ly * Lx, device=device, dtype=torch.complex64)
    start_idx, end_idx = 0, num_samples

    # Compile model with reduce-overhead mode
    compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

    # Warm-up
    for _ in range(5):
        _ = compiled_model(eta, G_eta, start_idx, end_idx)

    # Measure time with compiled model
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        eta = torch.randn(batch_size, Ltau * Ly * Lx, device=device, dtype=torch.complex64)
        G_eta = torch.randn(batch_size, Ltau * Ly * Lx, device=device, dtype=torch.complex64)
        y = compiled_model(eta, G_eta, start_idx, end_idx)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Compiled (reduce-overhead) model time: {end - start:.4f} sec")

    # Reset and measure eager time
    model_eager = InnerLoopModule(indices_r2, Ltau, Ly, Lx, inner_batch_size).to(device)
    for _ in range(5):  # also warm-up eager
        _ = model_eager(eta, G_eta, start_idx, end_idx)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        eta = torch.randn(batch_size, Ltau * Ly * Lx, device=device, dtype=torch.complex64)
        G_eta = torch.randn(batch_size, Ltau * Ly * Lx, device=device, dtype=torch.complex64)
        y = model_eager(eta, G_eta, start_idx, end_idx)

    torch.cuda.synchronize()
    end = time.time()

    print(f"Eager model time: {end - start:.4f} sec")

if __name__ == "__main__":
    test()