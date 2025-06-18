import numpy as np
import matplotlib.pyplot as plt
import os

Ls = [6, 10, 16, 20]
Ls = [L**3*40 for L in Ls]
latency_naive_gpu = [8.05, 13.52, 24.43, 30.09]
latency_opt_cuda = [2.71, 4.36, 8.41, 7.82]
speedup = [latency_naive_gpu[i] / latency_opt_cuda[i] for i in range(len(latency_naive_gpu))]   

fig, ax1 = plt.subplots()

# Plot latencies on the primary y-axis
ax1.plot(Ls, latency_naive_gpu, marker='o', linestyle='-', color='b', label='Naive GPU')
ax1.plot(Ls, latency_opt_cuda, marker='s', linestyle='-', color='g', label='Optimized CUDA kernel')
ax1.set_xlabel('(2+1D) lattice size (L^2 * Ltau)')
ax1.set_ylabel('Latency (s/sample)')
# ax1.set_title('Latency vs Ls')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a secondary y-axis for speedup
ax2 = ax1.twinx()
ax2.plot(Ls, speedup, marker='^', linestyle='--', color='r', label='Speedup (Naive/Opt)')
ax2.set_ylabel('Speedup')
ax2.legend(bbox_to_anchor=(0.42, 0.7))  # Shift down to avoid overlap

plt.show(block=False)

# Save the plot
script_path = os.path.dirname(os.path.abspath(__file__))
class_name = __file__.split('/')[-1].replace('.py', '')
method_name = "latency_speedup"
save_dir = os.path.join(script_path, f"./figures/{class_name}")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, f"{method_name}.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")


