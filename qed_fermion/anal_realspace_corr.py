import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat2 import initialize_coupling_mat

import matplotlib.pyplot as plt

# device = 

Lx = 16
Ly = 16
Ltau = 16
A = initialize_coupling_mat(Lx, Ly, Ltau, J=1, delta_tau=1)
A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
A = A.reshape([Ltau * Ly * Lx * 2, Ltau * Ly * Lx * 2])
# print(A[:9, :9])

greens = torch.linalg.inv(A)
greens = greens.view(Ltau, Ly, Lx, 2, Ltau, Ly, Lx, 2)
# greens = greens.permute([3, 2, 1, 0, 7, 6, 5, 4])

correlations = []
for dtau in range(Ltau):
    indices = range(Ltau - dtau)
    # corr = torch.mean(greens[indices, 0, 1, 0, [i + dtau for i in indices], 0, 1, 0])
    corr = torch.mean(greens[0, 0, 1, 0, dtau, 0, 1, 0])
    correlations.append(corr.item())

plt.figure()
plt.plot(correlations)
plt.show()

dbstop = 1

