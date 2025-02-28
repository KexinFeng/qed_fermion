import torch

shape = (8, 8)  # Example for 2D FFT
Lx, Ly = 1.0, 1.0  # Physical lengths of the domain

kx = torch.fft.fftfreq(shape[0], d=Lx / shape[0])
ky = torch.fft.fftfreq(shape[1], d=Ly / shape[1])

# Create a meshgrid for 2D wave numbers
KX, KY = torch.meshgrid(kx, ky, indexing='ij')

print(KX)
print(KY)


import torch

# Example values for k and L
k = torch.tensor([0.0, 1.0, 2.0, 3.0])  # Replace with your actual values
L = 4.0  # Replace with your actual value

# Compute the expression and unsqueeze
result = 2 * (1 - torch.cos(k * 2 * torch.pi / L))[None, None, None, None]

print(result.shape)
