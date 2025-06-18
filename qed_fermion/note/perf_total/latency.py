import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Data
L = [10, 12, 16, 20, 30, 36]  # beta = 2L
latency = [5.7, 7.4, 11.6, 20.8, 56.9, 94.8]  # sec/item

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
plt.loglog(L_cubed, latency, 'bo', label='Data')
plt.loglog(L_cubed, power_law(np.array(L_cubed), coeff, exponent), 'r-', 
           label=f'Fit: latency = {coeff:.2e} * (L³)^{exponent:.3f}')

plt.xlabel('L³')
plt.ylabel('Latency')
plt.title('Latency vs L³')
plt.legend()
plt.grid(True)
plt.show()

print(f"Scaling exponent for L³: {exponent:.3f}")
print(f"Alpha (latency ~ L^alpha): {alpha:.3f}")
print(f"Coefficient: {coeff:.2e}")

