import os
import matplotlib.pyplot as plt
import networkx as nx

# Create a 5x5 grid
G = nx.grid_2d_graph(5, 5)
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
            edge_colors[edge] = 'orange'

# Prepare edge color list in the order of G.edges()
edge_color_list = [edge_colors[tuple(sorted(edge))] for edge in G.edges()]

pos = {(x, y): (x, y) for x, y in G.nodes()}
nx.draw(G, pos, edge_color=edge_color_list, node_size=50, width=2)
plt.axis('equal')
# plt.show()

# Saver
file_name = "square_bond_family.pdf"
# Define save directory and file name
script_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_path, f"./figures")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, file_name)
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")
