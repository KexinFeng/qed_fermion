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


##
L = 3
F = torch.fft.fft(torch.eye(3), dim=0)
torch.eye(3) == 1/L * F.conj().T @ F

"""
torch.fft.fft(torch.eye(3), dim=0)
tensor([[ 1.0000+0.0000j,  1.0000+0.0000j,  1.0000+0.0000j],
        [ 1.0000+0.0000j, -0.5000-0.8660j, -0.5000+0.8660j],
        [ 1.0000-0.0000j, -0.5000+0.8660j, -0.5000-0.8660j]])

torch.fft.ifft(torch.eye(3), dim=0)
tensor([[ 0.3333-0.0000j,  0.3333-0.0000j,  0.3333-0.0000j],
        [ 0.3333-0.0000j, -0.1667+0.2887j, -0.1667-0.2887j],
        [ 0.3333+0.0000j, -0.1667-0.2887j, -0.1667+0.2887j]])
"""
