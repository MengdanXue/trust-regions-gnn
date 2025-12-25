"""
Graph Attention Network (GAT)
Velickovic et al., ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    2-layer Graph Attention Network

    Attention-based aggregation.
    Shows moderate U-shape pattern (3.2% amplitude).
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads,
                             heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels,
                             heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
