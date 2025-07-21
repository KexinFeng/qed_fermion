import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../../')
from qed_fermion.utils.prep_plots import set_default_plotting
set_default_plotting()

# Data
L = [10, 16, 20, 30, 36, 40, 46, 50]  # beta = L
se_latency = [0.13, 0.29, 0.91, 5.91, 11.75, 16.89, 32.15, 42.25]  # sec/item
mc_latency = [0.58, 0.85, 1.40, 2.98, 4.78, 5.64, 9.12, 11.79]  # sec/item
se_footage = [326.00, 552.00, 820.00, 2096.00, 3416.00, 4552.00, 7416.00, 8644.00]  # MB
mc_footage = [1683.69, 1949.69, 2145.69, 2647.69, 2979.69, 3187.69, 3547.69, 3791.69] # MB

latency = mc_latency
label = 'mc_latency'

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

# Add L^3^{7/3} guideline
# Choose a reference point to anchor the guideline for visual comparison
ref_idx = 0
ref_x = L_cubed[ref_idx]
ref_y = latency[ref_idx]
scaling = 7/3
guideline = ref_y * (np.array(L_cubed) / ref_x) ** scaling
plt.loglog(L_cubed, guideline, 'k--', label=r'$L^{7}$ guideline')

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
save_dir = os.path.join(script_path, "figures_Nrv40_plt")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, f"{label}_vs_L3_fit.pdf")
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

