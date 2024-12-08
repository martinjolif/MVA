"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
G = nx.read_edgelist("datasets/CA-HepTh.txt", comments='#', delimiter='\t')
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())


############## Task 2
print("Number of connected components: ", nx.number_connected_components(G))

giant_connected_component = max(nx.connected_components(G), key=len)
G_giant_connected_component = G.subgraph(giant_connected_component)

print("Number of edges of the giant connected component: ",
      G_giant_connected_component.number_of_edges(),
      ", it corresponds to :",
      100 * G_giant_connected_component.number_of_edges()/G.number_of_edges(),
      "% of the edges of the original graph.")

print("Number of nodes of the giant connected component: ",
      G_giant_connected_component.number_of_nodes(),
      ", it corresponds to :",
      100 * G_giant_connected_component.number_of_nodes() / G.number_of_nodes(),
      "% of the nodes of the original graph."
      )



