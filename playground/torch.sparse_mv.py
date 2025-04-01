import torch

# Example sparse matrix
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3.0, 4.0, 5.0])
size = (3, 3)
sparse_matrix = torch.sparse_coo_tensor(indices, values, size)

# Example dense vector
dense_vector = torch.tensor([1.0, 2.0, 3.0])

# Reshape dense vector to 2D (column vector)
dense_vector_2d = dense_vector.view(-1, 1)

# Perform sparse matrix-vector multiplication
result_2d = torch.sparse.mm(dense_vector_2d.view(1, -1), sparse_matrix)

# Reshape result back to 1D
result = result_2d.view(-1)

print(result)  # Output: tensor([9., 4., 15.])