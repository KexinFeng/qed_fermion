import torch
import time

# Define the size of the tensor
N = 10_000_000

# Benchmark torch.randn
torch.cuda.synchronize()
start = time.time()
randn_tensor = torch.randn(N, device='cuda')
torch.cuda.synchronize()
end = time.time()
print(f"torch.randn time: {end - start:.4f} seconds")

# Benchmark torch.randint
torch.cuda.synchronize()
start = time.time()
randint_tensor = torch.randint(0, 2, (N,), device='cuda')
torch.cuda.synchronize()
end = time.time()
print(f"torch.randint time: {end - start:.4f} seconds")
