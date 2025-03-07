import matlab.engine
import numpy as np


# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Create the sparse matrix A
A = eng.eval("sparse([1 1 2 2 3 3 4 4], [1 2 2 3 3 4 1 4], [5 -2 5 -2 5 2 -2 5])")

# Perform Cholesky factorization
L = eng.chol(A)

# Compute L'*L - A
diff = eng.eval("L'*L - A")

# Display sparse results
eng.eval("disp('--- sparse ----')", nargout=0)

# Perform incomplete Cholesky factorization
R = eng.ichol(A)

# Convert R to full matrix for display
R_full = eng.full(R)

# Compute R * R' - A
diff_sparse = eng.eval("R * R' - A")

# Compute full(R * R')
RR_full = eng.eval("full(R * R')")

# Compute full((R*R').*spones(A))
RR_sparse = eng.eval("full((R*R').*spones(A))")

# Compute (R*R').*spones(A) - A
diff_sparse_masked = eng.eval("(R*R').*spones(A) - A")

# Compute inv(R) * R
check = eng.eval("inv(R) * R")
check_full = eng.full(check)

# Convert results to Python objects if needed
R_full_py = eng.double(R_full)
check_full_py = eng.double(check_full)
