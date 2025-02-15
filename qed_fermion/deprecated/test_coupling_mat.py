import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.deprecated.coupling_mat import initialize_coupling_mat


# small lattice elementwise test
Lx = 2
Ly = 3
Ltau = 3
A = initialize_coupling_mat(Lx, Ly, Ltau, J=1, delta_tau=1)
A = A.to(torch.int32)

A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
A = A.reshape([Ltau * Ly * Lx * 2, Ltau * Ly * Lx * 2])
print(A[:14, :14])

A_expect = torch.tensor([[ 1, -1,  0,  1, -1,  0,  0,  0, -1],
        [-1,  1,  1, -1,  1,  0, -1,  0,  0],
        [ 0,  1,  1, -1,  0,  0, -1,  0,  0],
        [ 1, -1, -1,  1, -1,  0,  1,  0,  0],
        [-1,  1,  0, -1,  1, -1,  0,  1, -1],
        [ 0,  0,  0,  0, -1,  1,  1, -1,  1],
        [ 0, -1, -1,  1,  0,  1,  1, -1,  0],
        [ 0,  0,  0,  0,  1, -1, -1,  1, -1],
        [-1,  0,  0,  0, -1,  1,  0, -1,  1]], dtype=torch.int32)

A_expect = torch.tensor([
        [ 2, -1,  0,  1, -1,  0,  0,  0, -1],
        [-1,  2,  1, -2,  1,  0, -1,  0,  0],
        [ 0,  1,  2, -1,  0,  0, -1,  0,  0],
        [ 1, -2, -1,  2, -1,  0,  1,  0,  0],
        [-1,  1,  0, -1,  2, -1,  0,  1, -1],
        [ 0,  0,  0,  0, -1,  2,  1, -2,  1],
        [ 0, -1, -1,  1,  0,  1,  2, -1,  0],
        [ 0,  0,  0,  0,  1, -2, -1,  2, -1],
        [-1,  0,  0,  0, -1,  1,  0, -1,  2]], dtype=torch.int32)

# torch.testing.assert_close(A[:9, :9], A_expect)

# --------- symmetric and pos-def ---------
# Lx, Ly, Ltau = 2, 3, 2
# A = initialize_coupling_mat(Lx, Ly, Ltau, 0.5)
# # A = A.to(torch.int32)
# A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
# A = A.reshape([Ltau * Ly * Lx * 2, Ltau * Ly * Lx * 2])

Lx = 2
Ly = 3
Ltau = 3
A = initialize_coupling_mat(Lx, Ly, Ltau, J=1, delta_tau=1)

A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])
A = A.reshape([Ltau * Ly * Lx * 2, Ltau * Ly * Lx * 2])
print(A[:9, :9])

assert torch.allclose(A, A.T)

def is_positive_definite(A):
    # _ = torch.linalg.cholesky(A)
    eigenvalues = torch.linalg.eigvalsh(A)  # Only works for symmetric matrices
    print(eigenvalues)
    return torch.all(eigenvalues > 0)

# def is_positive_semidefinite(A, atol=1e-7):
eigenvalues = torch.linalg.eigvalsh(A)  # Optimized for symmetric matrices
print(eigenvalues)
atol = 1e-5
assert  torch.all(eigenvalues >= -atol)  # Allow small numerical errors
    
# assert is_positive_semidefinite(A)
