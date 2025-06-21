import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# Data
L = [10, 16, 20, 30, 36, 40, 46, 50]  # beta = L
se_latency = [0.09, 0.18, 0.37, 2.33, 5.66, 8.28, 15.38, 20.71]  # sec/item
mc_latency = [0.55, 0.85, 1.42, 2.93, 4.74, 5.53, 9.00, 11.84]  # sec/item
se_footage = [264.00, 376.00, 504.00, 1146.00, 1806.00, 2360.00, 3802.00, 4416.00]  # MB
mc_footage = [1683.69, 1949.69, 2145.69, 2647.69, 2979.69, 3187.69, 3547.69, 3791.69] # MB

latency = mc_footage
label = 'mc_footage MB'

# Convert L to L^3
L_cubed = [l**3 for l in L]

# Define power law function: y = a * x^b
def power_law(x, a, b):
    return a * np.power(x, b)

# Fit the power law
popt, pcov = curve_fit(power_law, L_cubed, latency)
coeff, exponent = popt

# Calculate alpha: since latency ~ L^alpha and we fit latency ~ (L^3)^exponent
# then latency ~ L^(3*exponent), so alpha = 3 * exponent
alpha = 3 * exponent

# Create plot
plt.figure(figsize=(8, 6))
plt.loglog(L_cubed, latency, 'bo')
plt.loglog(L_cubed, power_law(np.array(L_cubed), coeff, exponent), 'r-', 
           label=f'Fit: y = {coeff:.2e} * (L³)^{exponent:.3f}')

plt.xlabel('L^3')
plt.ylabel(label)
# plt.title('Latency vs L³')
plt.legend()
plt.grid(True)
# plt.show(blocking=False)

print(f"Scaling exponent for L³: {exponent:.3f}")
print(f"Alpha (latency ~ L^alpha): {alpha:.3f}")
print(f"Coefficient: {coeff:.2e}")

# Save the plot
script_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_path, "figures_Nrv20")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, f"{label}_vs_L3_fit.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

