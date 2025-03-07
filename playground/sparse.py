import matlab.engine 
import numpy as np
import scipy.sparse as sp

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the matrix A in Python (same as in MATLAB version)
A_numpy = np.array([[5, -2, 0, -2, -2],
                    [-2, 5, -2, 0, 0],
                    [0, -2, 5, -2, 0],
                    [-2, 0, -2, 5, -2],
                    [-2, 0, 0, -2, 5]])

# Convert the NumPy matrix to a sparse matrix using scipy
A_sparse = sp.csr_matrix(A_numpy)

# Convert sparse matrix to MATLAB sparse format
A_matlab = eng.sparse(A_sparse.indptr.tolist(), A_sparse.indices.tolist(), A_sparse.data.tolist(), A_sparse.shape[0], A_sparse.shape[1])

dbstop = 1
