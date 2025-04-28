import numpy as np
import matplotlib.pyplot as plt
import os

N = \
[1000, 2000, 5000, 1e4, 2e4, 4e4, 6e4]

convergence_steps = \
[15,   35,   85,   135, 190, 200, 205]

# Visualize the data
plt.figure(figsize=(8, 6))
plt.plot(N, convergence_steps, marker='o', linestyle='-', color='b', label='Convergence Steps')
plt.xscale('linear')
plt.xlabel('(2+1D) lattice size N', fontsize=12)
plt.ylabel('Convergence Steps', fontsize=12)
plt.title('pcg steps converged to 1e-3 ', fontsize=14)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# Save the plot
script_path = os.path.dirname(os.path.abspath(__file__))
class_name = __file__.split('/')[-1].replace('.py', '')
method_name = "convergence_steps"
save_dir = os.path.join(script_path, f"./figures/{class_name}")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, f"{method_name}.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")
