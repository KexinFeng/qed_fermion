import numpy as np
import torch 

# Lattice sizes
Nx, Ny, Nt = 8, 8, 5

# Define kx, ky, and w using discrete Fourier modes
kx_vals = 2 * torch.pi * torch.fft.fftfreq(Nx, d=1)
ky_vals = 2 * torch.pi * torch.fft.fftfreq(Ny, d=1)
w_vals = 2 * torch.pi * torch.fft.fftfreq(Nt, d=1)

# Create the meshgrid
kx, ky, w = torch.meshgrid(kx_vals, ky_vals, w_vals, indexing='ij')

# Print the shapes to confirm
print("kx shape:", kx.shape)
print("ky shape:", ky.shape)
print("w shape:", w.shape)

curl = lambda kx, ky, phi: phi[0] * (1 - torch.exp(1j*ky)) + phi[1] * (-1 + torch.exp(1j*kx)) 
moment = lambda kx, ky: torch.cos(kx) + torch.cos(ky) - 2

curl_vec = lambda kx, ky: torch.tensor([1 - torch.exp(1j*ky), -1 + torch.exp(1j*kx)]).unsqueeze(0)

Lmd = torch.zeros(Nt, Ny, Nx, 2)
cnt = 0
cnt2= 0
action = 0
for x, kx in enumerate(kx_vals):
    for y, ky in enumerate(ky_vals):
        for t, w in enumerate(w_vals):
            Gk = torch.tensor([[ 
                torch.cos(w) - 1 + torch.cos(kx) - 1,  
                1/2 * (1 - torch.exp(-1j * kx)) * (torch.exp(1j * ky) - 1)], 
                [1/2 * (torch.exp(1j*kx) - 1) * (1 - torch.exp(-1j*ky)), 
                torch.cos(w) - 1 + torch.cos(ky) - 1]])
            eigval, eigvec = torch.linalg.eigh(Gk)
            if (eigval.abs() < 1e-5).any():
                print(f'eig', eigval)
                idx = torch.nonzero(eigval.abs() < 1e-5).view(-1)
                print('idx', idx)
                print(f'w = {w}')
                idx_other = (idx + 1)%2
                print(f'kx, ky = {kx/torch.pi/2} {ky/torch.pi/2}')
                print(f'zero_mode: {eigvec[:, idx_other]}')
                for iother in idx_other.tolist():
                    print(f'curl = {curl(kx, ky, eigvec[:, iother])}')
                    print(f'local_curl, momentum= {moment(kx, ky)}')
                    if abs(curl(kx, ky, eigvec[:, iother])) > 0:
                        dbstop = 0
                print('\n')
                cnt += 1
                if (curl(kx, ky, eigvec[:, iother])).abs().item() < 1e-8:
                    cnt2 += 1

            eigval_clmp = torch.sign(eigval) * torch.clamp_min(eigval.abs(), 1e-8)
            
            eigval_factor = (1 - torch.cos(w)) * (torch.cos(w) - 1 + torch.cos(kx) + torch.cos(ky) - 2)
            elem = eigval_clmp ** (-1) * 1
            if (torch.isnan(elem)).any():
                dbstop = 0
            Lmd[t, y, x, :] = elem

print((Lmd.abs() < 1e-8).sum())

print((Lmd.abs() > 1e7).sum())

print(cnt, cnt2)
dbstop = 1
    
#### Find the energy of the mode #####

