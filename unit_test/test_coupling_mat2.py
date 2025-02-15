import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat2 import initialize_coupling_mat


# small lattice elementwise test
Lx = 2
Ly = 3
Ltau = 3
A = initialize_coupling_mat(Lx, Ly, Ltau, J=1, delta_tau=1)
A = A.to(torch.int32)

A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
A = A.reshape([Ltau * Ly * Lx * 2, Ltau * Ly * Lx * 2])
print(A[:12, :12])
A_expect = torch.tensor([[ 4, -1,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0],
        [-1,  4, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0],
        [ 0, -1,  4, -1,  0,  1, -1,  0,  0,  0,  0,  0],
        [ 1, -1, -1,  4, -1, -1,  1,  0,  0,  0,  0,  0],
        [-1,  1,  0, -1,  4, -1,  0,  1, -1,  0,  0,  0],
        [ 0,  0,  1, -1, -1,  4, -1, -1,  1,  0,  0,  0],
        [ 0,  0, -1,  1,  0, -1,  4, -1,  0,  1, -1,  0],
        [ 0,  0,  0,  0,  1, -1, -1,  4, -1, -1,  1,  0],
        [ 0,  0,  0,  0, -1,  1,  0, -1,  4, -1,  0,  1],
        [ 0,  0,  0,  0,  0,  0,  1, -1, -1,  4, -1, -1],
        [ 0,  0,  0,  0,  0,  0, -1,  1,  0, -1,  4, -1],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  4]], dtype=torch.int32)

torch.testing.assert_close(A[:12, :12], A_expect)

# --------- symmetric and pos-def ---------
Lx = 2
Ly = 3
Ltau = 3
A = initialize_coupling_mat(Lx, Ly, Ltau, J=1, delta_tau=1)

A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
A = A.reshape([Ltau * Ly * Lx * 2, Ltau * Ly * Lx * 2])
assert torch.allclose(A, A.T)

# is_positive_semidefinite
eigenvalues = torch.linalg.eigvalsh(A)  # Optimized for symmetric matrices
print(eigenvalues)
atol = 1e-5
assert  torch.all(eigenvalues >= -atol)  # Allow small numerical errors
    

