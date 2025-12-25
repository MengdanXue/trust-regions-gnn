"""
Cross-Model H-Sweep Experiment
Tests whether U-shape pattern holds across different GNN architectures.

This is the P0-1 "killer experiment" that proves U-shape is a data property,
not a model-specific artifact.

Models tested:
- MLP (baseline, no graph structure)
- GCN (mean aggregation)
- GAT (attention-based aggregation)
- GraphSAGE (sampling-based aggregation)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ============== Model Definitions ==============

class MLP(nn.Module):
    """Simple MLP baseline (no graph structure)"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

class GCN(nn.Module):
    """Graph Convolutional Network (mean aggregation)"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    """GraphSAGE with mean aggregation"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ============== Data Generation ==============

def generate_controlled_graph(n_nodes=1000, n_features=20, n_classes=2,
                               target_h=0.5, feature_separability=0.5, seed=42):
    """
    Generate a graph with controlled homophily and feature quality.
    """
    set_seed(seed)

    # Generate class labels (balanced)
    labels = np.array([i % n_classes for i in range(n_nodes)])
    np.random.shuffle(labels)

    # Generate features with controlled separability
    features = np.random.randn(n_nodes, n_features)
    class_centers = {}
    for c in range(n_classes):
        class_centers[c] = np.random.randn(n_features) * feature_separability * 2

    for i in range(n_nodes):
        features[i] += class_centers[labels[i]]

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Generate edges with target homophily
    n_edges = n_nodes * 15 // 2
    edge_list = []
    same_class_edges = 0
    diff_class_edges = 0

    target_same = int(n_edges * target_h)
    target_diff = n_edges - target_same

    # Generate same-class edges
    for _ in range(target_same * 3):
        if same_class_edges >= target_same:
            break
        c = np.random.randint(n_classes)
        class_nodes = np.where(labels == c)[0]
        if len(class_nodes) < 2:
            continue
        i, j = np.random.choice(class_nodes, 2, replace=False)
        if i != j and (i, j) not in edge_list and (j, i) not in edge_list:
            edge_list.append((i, j))
            same_class_edges += 1

    # Generate different-class edges
    for _ in range(target_diff * 3):
        if diff_class_edges >= target_diff:
            break
        c1 = np.random.randint(n_classes)
        c2 = (c1 + 1) % n_classes
        class1_nodes = np.where(labels == c1)[0]
        class2_nodes = np.where(labels == c2)[0]
        i = np.random.choice(class1_nodes)
        j = np.random.choice(class2_nodes)
        if (i, j) not in edge_list and (j, i) not in edge_list:
            edge_list.append((i, j))
            diff_class_edges += 1

    # Convert to edge_index format (undirected)
    edge_index = np.array(edge_list).T
    edge_index = np.hstack([edge_index, edge_index[[1, 0]]])

    actual_h = same_class_edges / (same_class_edges + diff_class_edges) if (same_class_edges + diff_class_edges) > 0 else 0

    return {
        'features': torch.tensor(features, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'h_target': target_h,
        'h_actual': actual_h,
        'n_nodes': n_nodes,
        'n_edges': edge_index.shape[1]
    }

# ============== Training ==============

def train_and_evaluate(model, data, epochs=200, lr=0.01):
    """Train model and return test accuracy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = data['features'].to(device)
    edge_index = data['edge_index'].to(device)
    y = data['labels'].to(device)
    n = len(y)

    # Create train/val/test split (60/20/20)
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:int(0.6*n)]] = True
    val_mask[perm[int(0.6*n):int(0.8*n)]] = True
    test_mask[perm[int(0.8*n):]] = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            pred = out.argmax(dim=1)

            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

    return best_test_acc

# ============== Main Experiment ==============

def run_cross_model_hsweep():
    """Run H-sweep experiment across multiple GNN architectures."""

    # Experiment parameters
    h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_runs = 5
    feature_separability = 0.5
    hidden_channels = 64

    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE
    }

    results = defaultdict(list)

    print("="*70)
    print("Cross-Model H-Sweep Experiment")
    print("Proving U-shape is a DATA property, not model-specific")
    print("="*70)
    print(f"Models: {list(model_classes.keys())}")
    print(f"H values: {h_values}")
    print(f"Runs per setting: {n_runs}")
    print("="*70)

    for h in h_values:
        print(f"\n[h = {h}]")
        model_accs = {name: [] for name in model_classes}

        for run in range(n_runs):
            seed = 42 + run * 100 + int(h * 1000)
            data = generate_controlled_graph(
                target_h=h,
                feature_separability=feature_separability,
                seed=seed
            )

            for model_name, model_class in model_classes.items():
                if model_name == 'GAT':
                    model = model_class(20, hidden_channels, 2, heads=4)
                else:
                    model = model_class(20, hidden_channels, 2)

                acc = train_and_evaluate(model, data)
                model_accs[model_name].append(acc)

        # Compute statistics
        h_result = {'h': h, 'h_actual': data['h_actual']}
        print(f"  Results:")
        for model_name in model_classes:
            mean_acc = np.mean(model_accs[model_name])
            std_acc = np.std(model_accs[model_name])
            h_result[f'{model_name}_acc'] = mean_acc
            h_result[f'{model_name}_std'] = std_acc
            print(f"    {model_name}: {mean_acc:.3f} +/- {std_acc:.3f}")

        # Compute advantages over MLP
        mlp_mean = h_result['MLP_acc']
        for model_name in ['GCN', 'GAT', 'GraphSAGE']:
            advantage = h_result[f'{model_name}_acc'] - mlp_mean
            h_result[f'{model_name}_advantage'] = advantage
            winner = model_name if advantage > 0 else 'MLP'
            print(f"    {model_name} vs MLP: {advantage:+.3f} [{winner}]")

        results['by_h'].append(h_result)

    # Summary analysis
    print("\n" + "="*70)
    print("SUMMARY: U-Shape Consistency Across Models")
    print("="*70)

    # Check if U-shape holds for each GNN model
    for model_name in ['GCN', 'GAT', 'GraphSAGE']:
        print(f"\n{model_name}:")
        advantages = [r[f'{model_name}_advantage'] for r in results['by_h']]
        h_vals = [r['h'] for r in results['by_h']]

        # Low h zone (h < 0.3)
        low_h_wins = sum(1 for r in results['by_h'] if r['h'] < 0.3 and r[f'{model_name}_advantage'] > 0)
        low_h_total = sum(1 for r in results['by_h'] if r['h'] < 0.3)

        # Mid h zone (0.3 <= h <= 0.7)
        mid_h_wins = sum(1 for r in results['by_h'] if 0.3 <= r['h'] <= 0.7 and r[f'{model_name}_advantage'] > 0)
        mid_h_total = sum(1 for r in results['by_h'] if 0.3 <= r['h'] <= 0.7)

        # High h zone (h > 0.7)
        high_h_wins = sum(1 for r in results['by_h'] if r['h'] > 0.7 and r[f'{model_name}_advantage'] > 0)
        high_h_total = sum(1 for r in results['by_h'] if r['h'] > 0.7)

        print(f"  Low h (<0.3): {model_name} wins {low_h_wins}/{low_h_total}")
        print(f"  Mid h (0.3-0.7): {model_name} wins {mid_h_wins}/{mid_h_total}")
        print(f"  High h (>0.7): {model_name} wins {high_h_wins}/{high_h_total}")

        # Check U-shape pattern
        u_shape = (low_h_wins >= low_h_total // 2) and (mid_h_wins <= mid_h_total // 2) and (high_h_wins >= high_h_total // 2)
        print(f"  U-shape pattern: {'YES' if u_shape else 'NO/PARTIAL'}")

    # Save results
    output = {
        'experiment': 'cross_model_hsweep',
        'models': list(model_classes.keys()),
        'h_values': h_values,
        'n_runs': n_runs,
        'feature_separability': feature_separability,
        'results': results['by_h']
    }

    output_path = Path(__file__).parent / "cross_model_hsweep_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output

if __name__ == "__main__":
    results = run_cross_model_hsweep()
