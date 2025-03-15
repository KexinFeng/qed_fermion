import torch
import time

def perform_operation(device, size):
    # Create large tensors
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start_time = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Perform matrix multiplication
    result = torch.matmul(a, b)
    
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time_cuda = start.elapsed_time(end)
    total_elapsed_time = time.perf_counter() - start_time
    return elapsed_time_cuda, total_elapsed_time

bs = 5
def perform_einsum(device, size):
    # Create large tensors
    a = torch.randn(bs, size, size, device=device)
    b = torch.randn(bs, size, size, device=device)
    
    start_time = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Perform einsum operation
    result = torch.einsum('bij,bjk->bik', a, b)
    
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time_cuda = start.elapsed_time(end)
    total_elapsed_time = time.perf_counter() - start_time
    return elapsed_time_cuda, total_elapsed_time

def perform_det(device, size):
    # Create large tensor
    a = torch.randn(bs, size, size, device=device)
    
    start_time = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Perform determinant calculation
    result = torch.linalg.det(a)
    
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time_cuda = start.elapsed_time(end)
    total_elapsed_time = time.perf_counter() - start_time
    return elapsed_time_cuda, total_elapsed_time

# Check if CUDA is available
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Set the devices
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CPU device: {cpu_device}")
print(f"CUDA device: {cuda_device}")


Lx, Ly, Ltau = 6, 6, 10
size = Lx * Ly * Ltau
sizes = [int(size * 0.1), int(size * 0.5), size, size * 5, size * 10, size*20]

for size in sizes:
    print(f"\nTesting with matrix size: {size}x{size}")
    
    cpu_einsum_time, _ = perform_einsum(cpu_device, size)
    cuda_einsum_time, total_einsum_time = perform_einsum(cuda_device, size)
    print(f"CPU einsum time: {cpu_einsum_time:.6f} seconds")
    print(f"CUDA einsum time: {cuda_einsum_time:.6f} ms")
    print(f"Total einsum time: {total_einsum_time:.6f} seconds")
    einsum_speedup = cpu_einsum_time / (cuda_einsum_time / 1000)
    print(f"Einsum speedup: {einsum_speedup:.2f}x\n")
    
    cpu_det_time, _ = perform_det(cpu_device, size)
    cuda_det_time, total_det_time = perform_det(cuda_device, size)
    print(f"CPU determinant time: {cpu_det_time:.6f} seconds")
    print(f"CUDA determinant time: {cuda_det_time:.6f} ms")
    print(f"Total determinant time: {total_det_time:.6f} seconds")
    det_speedup = cpu_det_time / (cuda_det_time / 1000)
    print(f"Determinant speedup: {det_speedup:.2f}x\n")
    
    cpu_time, _ = perform_operation(cpu_device, size)
    cuda_time, total_time = perform_operation(cuda_device, size)
    print(f"CPU matmul time: {cpu_time:.6f} seconds")
    print(f"CUDA matmul time: {cuda_time:.6f} ms")
    print(f"Total matmul time: {total_time:.6f} seconds")
    speedup = cpu_time / (cuda_time / 1000)
    print(f"Matmul speedup: {speedup:.2f}x")
    
    print('-----------------------------------')