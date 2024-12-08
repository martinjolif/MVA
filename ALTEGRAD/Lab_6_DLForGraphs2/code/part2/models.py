"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6
    
        ##################
        # your code here #
        ##################
        Z1 = self.relu(torch.mm(adj, self.fc1(x_in)))
        Z2 = self.relu(torch.mm(adj, self.fc2(Z1)))

        # zG = torch.sum(Z2, axis = 0, keepdim=True) # for batch size = 1

        # for batch_size > 1
        idx = idx.unsqueeze(1).repeat(1, Z2.size(1))
        zG = torch.zeros(torch.max(idx)+1, Z2.size(1)).to(self.device)
        zG = zG.scatter_add_(0, idx, Z2)
        
        ##################
        # your code here #
        ##################
        out = self.fc4(self.relu(self.fc3(zG)))

        return F.log_softmax(out, dim=1)
