"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    ##################
    # your code here #
    A = nx.adjacency_matrix(G)
    D_inv = diags([1 / G.degree(node) for node in G.nodes()], format='csr')
    I = eye(G.number_of_nodes(), format='csr')

    L = I - D_inv @ A
    eigenvalues, eigenvectors = eigs(L, k, which='SR')  # get the smallest eigenvalues
    U = eigenvectors.real

    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(U)

    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}
    ##################

    return clustering


############## Task 4

##################
# your code here #
G = nx.read_edgelist("datasets/CA-HepTh.txt", comments='#', delimiter='\t')
giant_connected_component = max(nx.connected_components(G), key=len)
giant_connected_component_graph = G.subgraph(giant_connected_component)
clustering = spectral_clustering(giant_connected_component_graph, 50)
##################



############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    nc = list(set(clustering.values()))
    m = G.number_of_edges()
    modularity = 0
    for n in nc:
        nodes_in_cluster_n = [node for node, cluster in clustering.items() if cluster == n]
        subgraph_cluster = G.subgraph(nodes_in_cluster_n)
        l = subgraph_cluster.number_of_edges()

        d = 0
        for node in nodes_in_cluster_n:
            d += G.degree(node)

        modularity += l / m - (d / (2 * m)) ** 2
    ##################
    
    return modularity

############## Task 6

##################
# your code here #
##################

modularity_clustering = modularity(giant_connected_component_graph, clustering)
print("Modularity obtained through Spectral Clustering algorithm (k=50): ", modularity_clustering)

clustering_random = {node: randint(0,49) for i, node in enumerate(giant_connected_component_graph.nodes())}
modularity_clustering_random = modularity(giant_connected_component_graph, clustering_random)
print("Modularity obtained by randomly partition the nodes into 50 clusters: ", modularity_clustering_random)


