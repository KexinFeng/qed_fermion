import numpy as np
import scipy.sparse as sp
import pyamg  # Install via: pip install pyamg

# Define an 8x8 block diagonal matrix (2x2 blocks)
block = np.array([[5, -2], [-2, 5]])
A_blocks = [block] * 4  # Four repeated 2x2 blocks
A_sparse = sp.block_diag(A_blocks, format='csr')  # Sparse 8x8 matrix

print("A (Original Sparse Matrix):\n", A_sparse.toarray())

# Compute Incomplete Cholesky Factorization using PyAMG
ml = pyamg.smoothed_aggregation_solver(A_sparse)  # Multigrid-based IC approximation
M = ml.aspreconditioner(cycle='V')  # Extract the preconditioner

# Convert preconditioner to a sparse matrix (acts like ichol(A))
R = sp.csr_matrix(M.matmat(np.eye(A_sparse.shape[0])))  # Extract approximate factor

print("\nR (Sparse Incomplete Cholesky Factor):\n", R.toarray())

# Compute R * R.T
RRT = R @ R.T

print("\nR * R.T (Reconstructed Matrix):\n", RRT.toarray())

# Compute the difference: Full RRT vs. A
diff_full = RRT - A_sparse
print("\nDifference (R * R.T - A):\n", diff_full.toarray())

# Step 1: Get the sparsity pattern of A (like MATLAB `spones(A)`)
A_sparsity_pattern = A_sparse.copy()
A_sparsity_pattern.data[:] = 1  # Convert all nonzero entries to 1

# Step 2: Apply the sparsity pattern explicitly
RRT_sparsity_preserved = RRT.multiply(A_sparsity_pattern)

print("\nRRT with A's Sparsity Structure Applied:\n", RRT_sparsity_preserved.toarray())

# Compute the sparsity-preserved difference
diff_sparse_preserved = RRT_sparsity_preserved - A_sparse
print("\nSparsity-Preserved Difference ((R @ R.T) .* A - A):\n", diff_sparse_preserved.toarray())
