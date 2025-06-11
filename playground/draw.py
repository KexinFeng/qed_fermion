import matplotlib.pyplot as plt
import networkx as nx

# Create a 5x5 grid
G = nx.grid_2d_graph(5, 5)
edge_colors = []

for (u, v) in G.edges():
    x1, y1 = u
    x2, y2 = v
    if x1 == x2:  # Vertical edge
        color = 'red' if x1 % 2 == 0 else 'blue'
    else:  # Horizontal edge
        color = 'green' if y1 % 2 == 0 else 'orange'
    edge_colors.append(color)

pos = {(x, y): (x, y) for x, y in G.nodes()}
nx.draw(G, pos, edge_color=edge_colors, node_size=50, width=2)
plt.axis('equal')
plt.show()
