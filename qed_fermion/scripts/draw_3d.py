import os
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
import os
rcParams['figure.raise_window'] = False


# Create a 3x3 grid
G = nx.grid_2d_graph(3, 3)
edge_colors = {}

# Assign default color to all edges
for edge in G.edges():
    edge_colors[edge] = 'gray'

# Color the four bonds around each (i+j)-even node
for (i, j) in G.nodes():
    if (i + j) % 2 == 0:
        # Right bond
        if (i, j + 1) in G:
            edge = tuple(sorted([ (i, j), (i, j + 1) ]))
            edge_colors[edge] = 'red'
        # Up bond
        if (i - 1, j) in G:
            edge = tuple(sorted([ (i, j), (i - 1, j) ]))
            edge_colors[edge] = 'blue'
        # Left bond
        if (i, j - 1) in G:
            edge = tuple(sorted([ (i, j), (i, j - 1) ]))
            edge_colors[edge] = 'green'
        # Down bond
        if (i + 1, j) in G:
            edge = tuple(sorted([ (i, j), (i + 1, j) ]))
            edge_colors[edge] = "purple"

# Prepare edge color list in the order of G.edges()
edge_color_list = [edge_colors[tuple(sorted(edge))] for edge in G.edges()]

pos = {(x, y): (x, y) for x, y in G.nodes()}
nx.draw(G, pos, edge_color=edge_color_list, node_size=100, width=2)

# Add orange circles at the midpoint of each edge
for edge in G.edges():
    (x0, y0) = pos[edge[0]]
    (x1, y1) = pos[edge[1]]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    plt.scatter(mx, my, s=80, facecolors='orange', edgecolors='orange', linewidths=2, zorder=3)

plt.axis('equal')
# plt.show(block=False)

dbstop = 1

file_name = "square_bond_family.pdf"
# Define save directory and file name
script_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_path, f"./figures")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, file_name)
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")
