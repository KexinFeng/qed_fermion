import os
import psutil
import torch

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 ** 3):.2f} GB")

print_memory_usage()  # Before tensor allocation
tensor = torch.ones((1024, 1024, 512), dtype=torch.float32, device="mps")
print_memory_usage()  # After tensor allocation
