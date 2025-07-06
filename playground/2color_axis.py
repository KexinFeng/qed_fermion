import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = 0.1 * x**2

# Create the figure and the first axis
fig, ax1 = plt.subplots()

# Plot data on ax1 (left y-axis)
color1 = 'blue'
ax1.plot(x, y1, color=color1)
ax1.set_ylabel('sin(x)', color=color1)
ax1.tick_params(axis='y', colors=color1)
ax1.yaxis.label.set_color(color1)

# Ensure left spine is visible and colored
ax1.spines['left'].set_color(color1)
ax1.spines['left'].set_linewidth(2)
ax1.spines['left'].set_visible(True)

# Create the twin axis sharing the x-axis
ax2 = ax1.twinx()

# Plot data on ax2 (right y-axis)
color2 = 'red'
ax2.plot(x, y2, color=color2, linestyle='--')
ax2.set_ylabel('0.1 * x^2', color=color2)
ax2.tick_params(axis='y', colors=color2)
ax2.yaxis.label.set_color(color2)

# Ensure right spine is visible and colored
ax2.spines['right'].set_color(color2)
ax2.spines['right'].set_linewidth(2)
ax2.spines['right'].set_visible(True)

# Optional: Hide top and bottom if needed
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_color('black')

# Final touches
ax1.set_xlabel('x')
plt.title('Twin Axes with Visible Colored Spines')

plt.tight_layout()
plt.show()
