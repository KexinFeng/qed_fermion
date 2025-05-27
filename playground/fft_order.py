import torch

def reorder_fft_grid(tensor2d):
    """Reorder the last two axes of a tensor from FFT-style to ascending momentum order."""
    Ny, Nx = tensor2d.shape[-2], tensor2d.shape[-1]
    return torch.roll(tensor2d, shifts=(Ny // 2, Nx // 2), dims=(-2, -1))

# Set grid size
Ny, Nx = 6, 6

# Create a 2D grid of momentum coordinates (as from fftfreq)
ky = torch.fft.fftfreq(Ny)  # e.g., [0, 1/6, 2/6, -3/6, -2/6, -1/6]
kx = torch.fft.fftfreq(Nx)

# Form 2D meshgrid
Ks = torch.stack(torch.meshgrid(ky, kx, indexing='ij'), dim=-1)  # KY.shape = [Ny, Nx]

# # Some dummy data aligned with FFT momentum grid (e.g., spsm_k)
# spsm_k = torch.arange(Ny * Nx).view(Ny, Nx).float()

# Print before reordering
print("Original ky row 0:")
print(Ks)
print("Original spsm_k:")
# print(spsm_k)

# Reorder using the function
KY_reordered = reorder_fft_grid(Ks.permute(2, 0, 1)).permute(1, 2, 0)
# spsm_k_reordered = reorder_fft_grid(spsm_k)

# Print after reordering
print("\nReordered ky row 0:")
print(KY_reordered)
print("Reordered spsm_k:")
# print(spsm_k_reordered)
