import torch

# Create a batch of square matrices
A = torch.randn(3, 2, 2)

# Compute the determinant of each matrix in the batch
determinants = torch.linalg.det(A)

print(determinants)
print(determinants.shape)


# =======================================================================

import torch
# Set the device and dtype
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
dtype = torch.float64

# Create a batch of positive definite matrices
batch_size = 3
matrix_size = 4
A = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=dtype)
A = torch.matmul(A, A.transpose(-1, -2))  # Make the matrices positive definite

# Perform Cholesky decomposition on the batch of matrices
L = torch.linalg.cholesky(A)

# Compute the inverse of the matrices using the Cholesky decomposition
A_inv = torch.cholesky_inverse(L)

# Verify the result by multiplying the original matrices with their inverses
I = torch.matmul(A, A_inv)

# Print the results
print("Original matrices (A):")
print(A)
print("\nCholesky decomposition (L):")
print(L)
print("\nInverse matrices (A_inv):")
print(A_inv)
print("\nProduct of original and inverse matrices (should be identity):")
print(I)