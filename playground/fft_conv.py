import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Set dimensions
Lx, Ly, Ltau = 4, 4, 4
N = Lx * Ly * Ltau
shape = (Ltau, Ly, Lx)

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# Generate sample inputs
Nrv = 10 # Number of random vectors
a = torch.randn((Nrv,) + shape, dtype=torch.cfloat, device=device)  # [1, Ltau, Ly, Lx]
b = torch.randn((Nrv,) + shape, dtype=torch.cfloat, device=device)

# Flatten: [1, Ltau, Ly, Lx] -> [1, N]
a_flat = a.reshape(Nrv, -1)
b_flat = b.reshape(Nrv, -1)

# ---- LHS: Real-space sum over r ----
G = torch.einsum('bi,bj->ji', a_flat, b_flat) / a_flat.shape[0]  # [N, N]

result = torch.empty((N, N), dtype=G.dtype, device=device)
for i in range(N):
    for d in range(N):
        tau = i % Ltau
        y = (i // Ltau) % Ly
        x = i // (Ltau * Ly)

        dtau = d % Ltau
        dy = (d // Ltau) % Ly
        dx = d // (Ltau * Ly)

        idx = ((tau + dtau) % Ltau) + ((y + dy) % Ly) * Ltau + ((x + dx) % Lx) * (Ltau * Ly)
        result[i, d] = G[idx, i]
G_mean_LHS = result.mean(dim=0).view(Ltau, Ly, Lx).permute(2, 1, 0)  # [Lx, Ly, Ltau]

# ---- RHS: Fourier convolution ----
a_F_neg_k = torch.fft.ifftn(a, dim=(-3, -2, -1), norm="backward")
b_F = torch.fft.fftn(b, dim=(-3, -2, -1), norm="forward")

G_delta_0_RHS = torch.fft.ifftn(a_F_neg_k * b_F, dim=(-3, -2, -1), norm="forward").mean(dim=0)  # [Ltau, Ly, Lx]
G_delta_0_RHS = G_delta_0_RHS.permute(2, 1, 0)  # [Lx, Ly, Ltau]

# ---- Comparison ----
torch.testing.assert_close(G_delta_0_RHS.real, G_mean_LHS.real, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(G_delta_0_RHS.imag, G_mean_LHS.imag, rtol=1e-4, atol=1e-4)

print("✅ Both implementations match!")
