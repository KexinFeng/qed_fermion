import torch

# Define indices and values for the COO tensor
indices = torch.tensor([[0, 0, 1], [1, 2, 2]])
values = torch.tensor([3, 4, 5])
size = (2, 3)

# Create the sparse COO tensor
coo = torch.sparse_coo_tensor(indices, values, size)

# Coalesce the COO tensor to ensure it's in a standard form
coo = coo.coalesce()

# Retrieve the indices and values from the COO tensor
retrieved_indices = coo.indices()
retrieved_values = coo.values()

print("COO Indices:", retrieved_indices)
print("COO Values:", retrieved_values)

# Convert the coalesced COO tensor to a sparse CSR tensor
csr_tensor = coo.to_sparse_csr()

# Accessing the components
retrieved_crow_indices = csr_tensor.crow_indices()
retrieved_col_indices = csr_tensor.col_indices()
retrieved_values = csr_tensor.values()

print("Compressed Row Indices (crow_indices):", retrieved_crow_indices)
print("Column Indices (col_indices):", retrieved_col_indices)
print("Values:", retrieved_values)

# Convert the CSR tensor back to a dense tensor and print it
dense_tensor = csr_tensor.to_dense()
print("CSR Tensor in full matrix form:")
print(dense_tensor)
