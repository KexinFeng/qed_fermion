import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Example tensors (Move to GPU if needed)
n, m = 2, 2  # Dimensions of B
tau = 2  # Block index
tau_max = 3

M = torch.eye(m * tau_max, device=device)  # Large matrix
B = torch.randn(n, m, device=device)  # Block data

# Compute block indices
row_start = n * ((tau + 1)%tau_max)
row_end = n * ((tau + 2))
col_start = m * tau
col_end = m * ((tau + 1)%tau_max)

# Inplace assignment
if tau < tau_max - 1:
    row_start = n * (tau + 1)
    row_end = n * (tau + 2)
    col_start = n * tau
    col_end = n * (tau + 1)
    M[row_start:row_end, col_start:col_end] = -B
else:
    M[:n, n*tau:] = B

print(M)
