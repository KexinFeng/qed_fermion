import matplotlib.pyplot as plt
import numpy as np

n = 30  # Number of distinct colors
colors = plt.cm.get_cmap("hsv", n)
colors = plt.cm.Set1(range(n))
colors = plt.cm.tab20(range(n))

for i in range(n):
    plt.plot([0, 1], [i, i], color=colors[i], lw=4)

plt.show()
