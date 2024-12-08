"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():

    ##################
    # your code here #
    dataset = TUDataset('datasets', name='MUTAG')

    # Convert each graph in the dataset to NetworkX format
    Gs = [to_networkx(data, to_undirected=True) for data in dataset]
    ##################
    # Extract labels for each graph
    y = [data.y.item() for data in dataset]

    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))

    ##################
    # your code here #
    for i in range(len(G_train)):
        for _ in range(n_samples):
            nodes = np.random.choice(list(G_train[i].nodes()), 3)
            subgraph = G_train[i].subgraph(nodes)
            if nx.is_isomorphic(subgraph, graphlets[0]):
                phi_train[i,0] += 1
            elif nx.is_isomorphic(subgraph, graphlets[1]):
                phi_train[i,1] += 1
            elif nx.is_isomorphic(subgraph, graphlets[2]):
                phi_train[i,2] += 1
            elif nx.is_isomorphic(subgraph, graphlets[3]):
                phi_train[i,3] += 1

    ##################

    phi_test = np.zeros((len(G_test), 4))
    
    ##################
    # your code here #
    for i in range(len(G_test)):
        for _ in range(n_samples):
            nodes = np.random.choice(list(G_test[i].nodes()), 3)
            subgraph = G_test[i].subgraph(nodes)
            if nx.is_isomorphic(subgraph, graphlets[0]):
                phi_test[i,0] += 1
            elif nx.is_isomorphic(subgraph, graphlets[1]):
                phi_test[i,1] += 1
            elif nx.is_isomorphic(subgraph, graphlets[2]):
                phi_test[i,2] += 1
            elif nx.is_isomorphic(subgraph, graphlets[3]):
                phi_test[i,3] += 1
    ##################

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9

##################
# your code here #
##################
K_train_gk, K_test_gk = graphlet_kernel(G_train, G_test)



############## Task 10

##################
# your code here #
##################
# Initialize SVM and train
clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)
# Predict
y_pred_sp = clf_sp.predict(K_test_sp)
print("Classification accuracy of the SVM with shortest path kernel: ", accuracy_score(y_test, y_pred_sp))

clf_gk = SVC(kernel='precomputed')
clf_gk.fit(K_train_gk, y_train)
y_pred_gk = clf_gk.predict(K_test_gk)
print("Classification accuracy of the SVM with graphlet kernel: ", accuracy_score(y_test, y_pred_gk))

