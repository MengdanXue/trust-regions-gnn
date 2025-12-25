"""
Metrics for Trust Regions Framework
"""

import numpy as np
import torch


def calculate_spi(h):
    """
    Calculate Structural Predictability Index.

    SPI = |2h - 1|

    Interpretation:
    - h = 0.5 -> SPI = 0 (no structural signal, maximum uncertainty)
    - h = 0 or h = 1 -> SPI = 1 (maximum structural signal)

    Args:
        h: Edge homophily ratio (0 to 1)

    Returns:
        SPI value (0 to 1)
    """
    return abs(2 * h - 1)


def calculate_edge_homophily(edge_index, labels):
    """
    Calculate edge homophily ratio.

    h = |{(u,v) in E : y_u = y_v}| / |E|

    Args:
        edge_index: [2, num_edges] tensor of edge indices
        labels: [num_nodes] tensor of node labels

    Returns:
        Homophily ratio (0 to 1)
    """
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def calculate_node_homophily(edge_index, labels):
    """
    Calculate node-level homophily (average per-node homophily).

    h_node = (1/|V|) * sum_v (|{u in N(v) : y_u = y_v}| / |N(v)|)

    Args:
        edge_index: [2, num_edges] tensor
        labels: [num_nodes] tensor

    Returns:
        Node homophily ratio (0 to 1)
    """
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    num_nodes = labels.size(0)
    src, dst = edge_index[0], edge_index[1]

    node_homophily = torch.zeros(num_nodes)
    node_degree = torch.zeros(num_nodes)

    for i in range(edge_index.size(1)):
        u, v = src[i].item(), dst[i].item()
        node_degree[u] += 1
        if labels[u] == labels[v]:
            node_homophily[u] += 1

    # Avoid division by zero
    mask = node_degree > 0
    node_homophily[mask] /= node_degree[mask]

    return node_homophily[mask].mean().item()


def should_use_gnn(h, threshold=0.67):
    """
    Decision rule: Should we use GNN based on homophily?

    Trust Region: SPI > threshold -> use GNN
    Uncertainty Zone: SPI <= threshold -> use MLP

    Args:
        h: Edge homophily ratio
        threshold: SPI threshold (default 0.67)

    Returns:
        True if GNN recommended, False if MLP recommended
    """
    spi = calculate_spi(h)
    return spi > threshold


def get_trust_region(h):
    """
    Classify homophily into trust regions.

    Args:
        h: Edge homophily ratio

    Returns:
        String: 'high_trust', 'low_trust', or 'uncertainty'
    """
    if h > 0.7:
        return 'high_trust'
    elif h < 0.3:
        return 'low_trust'
    else:
        return 'uncertainty'


def calculate_gnn_advantage(gnn_acc, mlp_acc):
    """
    Calculate GNN advantage over MLP.

    Args:
        gnn_acc: GNN accuracy (0 to 1 or percentage)
        mlp_acc: MLP accuracy (0 to 1 or percentage)

    Returns:
        Advantage (positive = GNN better, negative = MLP better)
    """
    return gnn_acc - mlp_acc
