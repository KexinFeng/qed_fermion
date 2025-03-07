import matlab.engine
import numpy as np


# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the sparse matrix A in Python (8x8 block diagonal)
A_numpy = np.array([[5, -2,  0, -2, -2],
                    [-2,  5, -2,  0,  0],
                    [ 0, -2,  5, -2,  0],
                    [-2,  0, -2,  5, -2],
                    [-2,  0,  0, -2,  5]])

# Convert A to MATLAB sparse format
A_matlab = matlab.double(A_numpy.tolist())  # Convert to MATLAB double

# Call MATLAB's ichol() function
R_matlab = eng.ichol(eng.sparse(A_matlab))

# Convert the result back to a NumPy array
R_numpy = np.array(R_matlab)

# Print results
print("A (Original Sparse Matrix):\n", A_numpy)
print("\nR (MATLAB Incomplete Cholesky Factor):\n", R_numpy)

# Compute R * R.T in Python
RRT = R_numpy @ R_numpy.T

print("\nR * R.T (Reconstructed Matrix):\n", RRT)

# Compute the difference R * R.T - A
diff_full = RRT - A_numpy
print("\nDifference (R * R.T - A):\n", diff_full)

# Compute sparsity-preserved difference
A_sparsity_pattern = (A_numpy != 0).astype(int)  # Equivalent to MATLAB's spones(A)
RRT_sparsity_preserved = RRT * A_sparsity_pattern

print("\nRRT with A's Sparsity Structure Applied:\n", RRT_sparsity_preserved)

diff_sparse_preserved = RRT_sparsity_preserved - A_numpy
print("\nSparsity-Preserved Difference ((R @ R.T) .* A - A):\n", diff_sparse_preserved)

# Close MATLAB engine
eng.quit()
