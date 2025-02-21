import torch

# Define the shape of the tensor
Lx, Ly, Ltau = 10, 10, 20  # Example sizes
shape = (Lx, Ly, Ltau)

# Example linear index (can be a tensor of indices)
idx = torch.tensor([123])  # Example linear index

# Convert linear index to sub-indices
sub_indices = torch.unravel_index(idx, shape)

# Print result
print(sub_indices)  # (tensor([...]), tensor([...]), tensor([...]))
