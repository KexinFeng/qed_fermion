import torch
import time

def perform_operation(device, size):
    # Create large tensors
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start_time = time.time()
    
    # Perform matrix multiplication
    result = torch.matmul(a, b)
    
    end_time = time.time()
    return end_time - start_time

# Check if MPS (Metal Performance Shaders) is available
print(f"Is MPS (Metal Performance Shaders) available? {torch.cuda.is_available()}")

# Set the devices
cpu_device = torch.device("cpu")
mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CPU device: {cpu_device}")
print(f"CUDA device: {mps_device}")

# Test with different matrix sizes
sizes = [1000, 2000, 3000, 5000]

for size in sizes:
    print(f"\nTesting with matrix size: {size}x{size}")
    
    cpu_time = perform_operation(cpu_device, size)
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    mps_time = perform_operation(mps_device, size)
    print(f"CUDA time: {mps_time:.4f} seconds")
    
    speedup = cpu_time / mps_time
    print(f"Speedup: {speedup:.2f}x")

