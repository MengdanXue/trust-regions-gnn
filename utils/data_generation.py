"""
Synthetic Graph Generation with Controlled Homophily
"""

import numpy as np
import torch
from torch_geometric.data import Data


def generate_csbm_graph(n_nodes=1000, n_features=20, n_classes=2,
                        target_h=0.5, feature_separability=0.5, seed=42):
    """
    Generate a Contextual Stochastic Block Model graph with controlled homophily.

    This allows systematic study of GNN behavior across the homophily spectrum.

    Args:
        n_nodes: Number of nodes
        n_features: Feature dimension
        n_classes: Number of classes (default 2 for binary classification)
        target_h: Target edge homophily (0 to 1)
        feature_separability: How separable features are (0=overlapping, 1=perfectly separable)
        seed: Random seed for reproducibility

    Returns:
        PyG Data object with:
            - x: Node features [n_nodes, n_features]
            - edge_index: Edge connectivity [2, n_edges]
            - y: Node labels [n_nodes]
            - h_actual: Actual achieved homophily
    """
    np.random.seed(seed)

    # Generate balanced class labels
    labels = np.zeros(n_nodes, dtype=np.int64)
    labels[n_nodes // 2:] = 1

    # Generate class-specific feature centers
    # feature_separability controls how well MLP can classify
    center_distance = feature_separability * 2.0
    centers = np.zeros((n_classes, n_features))
    centers[0] = -center_distance / 2
    centers[1] = center_distance / 2

    # Generate features with noise
    noise_std = 1.0  # Fixed noise level
    features = np.zeros((n_nodes, n_features))
    for i in range(n_nodes):
        features[i] = centers[labels[i]] + np.random.randn(n_features) * noise_std

    # Generate edges with controlled homophily
    avg_degree = 15
    n_edges_target = n_nodes * avg_degree // 2

    # Compute connection probabilities to achieve target homophily
    # p_same = probability of connecting same-class nodes
    # p_diff = probability of connecting different-class nodes
    # h = (n_same * p_same) / (n_same * p_same + n_diff * p_diff)

    n_same_pairs = (n_nodes // 2) ** 2 * 2  # Within each class
    n_diff_pairs = (n_nodes // 2) ** 2 * 2  # Between classes

    # Solve for p_same and p_diff given target_h
    if target_h > 0.5:
        p_same = 1.0
        p_diff = (1 - target_h) / target_h
    else:
        p_diff = 1.0
        p_same = target_h / (1 - target_h) if target_h > 0 else 0

    # Normalize to achieve target edge count
    scale = n_edges_target / (n_same_pairs * p_same + n_diff_pairs * p_diff) * 2
    p_same *= scale
    p_diff *= scale

    # Cap probabilities
    p_same = min(p_same, 0.5)
    p_diff = min(p_diff, 0.5)

    # Generate edges
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if labels[i] == labels[j]:
                if np.random.random() < p_same:
                    edges.append([i, j])
                    edges.append([j, i])
            else:
                if np.random.random() < p_diff:
                    edges.append([i, j])
                    edges.append([j, i])

    edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)

    # Calculate actual homophily
    if edge_index.shape[1] > 0:
        src, dst = edge_index
        same_class = (labels[src] == labels[dst])
        h_actual = same_class.mean()
    else:
        h_actual = 0.5

    # Convert to PyTorch tensors
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)
    y = torch.LongTensor(labels)

    # Create masks (60/20/20 split)
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)

    perm = np.random.permutation(n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.h_actual = h_actual

    return data


def generate_h_sweep_datasets(h_values, n_nodes=1000, n_features=20,
                               feature_separability=0.5, n_seeds=5):
    """
    Generate datasets across a range of homophily values.

    Args:
        h_values: List of target homophily values
        n_nodes: Nodes per graph
        n_features: Feature dimension
        feature_separability: Feature quality
        n_seeds: Number of random seeds per h value

    Returns:
        List of (h_target, seed, Data) tuples
    """
    datasets = []
    for h in h_values:
        for seed in range(n_seeds):
            data = generate_csbm_graph(
                n_nodes=n_nodes,
                n_features=n_features,
                target_h=h,
                feature_separability=feature_separability,
                seed=seed * 1000 + int(h * 100)
            )
            datasets.append((h, seed, data))
    return datasets
