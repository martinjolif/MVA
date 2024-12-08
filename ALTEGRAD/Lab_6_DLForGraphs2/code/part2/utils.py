"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()

    ############## Task 5
    
    ##################
    # your code here #
    ##################
    p1 = 0.2
    p2 = 0.4

    for k in range(50):
        n1 = randint(10, 20)
        Gs.append(nx.Graph(nx.fast_gnp_random_graph(n1, p1)))
        y.append(0)

        n2 = randint(10, 20)
        Gs.append(nx.Graph(nx.fast_gnp_random_graph(n2, p2)))
        y.append(1)

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
