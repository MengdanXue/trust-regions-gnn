"""
MLP Baseline (no graph structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    2-layer Multi-Layer Perceptron

    Baseline that ignores graph structure.
    Performs well when h is in [0.3, 0.7] (uncertainty zone).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # edge_index is ignored (for API compatibility)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
