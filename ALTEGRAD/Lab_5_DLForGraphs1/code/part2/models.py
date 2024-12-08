"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, return_message=False):
        ############## Tasks 10 and 13
        out_layer_1 = self.fc1(x_in)
        Z0 = self.dropout(self.relu(adj @ out_layer_1))
        out_layer_2 = self.fc2(Z0)
        Z1 = self.relu(adj @ out_layer_2)
        x = self.fc3(Z1)

        if return_message:
            return F.log_softmax(x, dim=1), Z1
        else:
            return F.log_softmax(x, dim=1)
