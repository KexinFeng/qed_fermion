import numpy as np
import matplotlib.pyplot as plt

Ls = [16, 20, 24]
mem_usage = [8, 15, 26]

# Compute Ls**3
Ls_cubed = [L**3 for L in Ls]

# Predict memory usage for L = 30 using linear extrapolation
L_new = 30
L_new_cubed = L_new**3
coefficients = np.polyfit(Ls_cubed, mem_usage, 1)  # Fit a linear model
mem_usage_new = np.polyval(coefficients, L_new_cubed)

# Add the new data point
Ls_cubed.append(L_new_cubed)
mem_usage.append(mem_usage_new)

# Create the plot
plt.plot(Ls_cubed[:-1], mem_usage[:-1], marker='o', linestyle='-', color='b', label='Memory Usage')
plt.scatter([L_new_cubed], [mem_usage_new], color='r', label=f'Prediction for L={L_new}')
plt.plot([Ls_cubed[-2], L_new_cubed], [mem_usage[-2], mem_usage_new], linestyle='--', color='b')

# Add labels and title
plt.xlabel('Ls')
plt.ylabel('Memory Usage (GB)')
plt.title('Memory Usage vs Ls^3')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


