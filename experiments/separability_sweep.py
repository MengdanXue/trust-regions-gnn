"""
Feature Separability Sweep Experiment
Tests whether the U-shape pattern holds across different feature quality levels.

This addresses Codex's concern: "U-shape might be artifact of synthetic setup"
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path
from collections import defaultdict

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class GCN(nn.Module):
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

class MLP(nn.Module):
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

def generate_controlled_graph(n_nodes=1000, n_features=20, n_classes=2,
                               target_h=0.5, feature_separability=0.5, seed=42):
    """
    Generate a graph with controlled homophily and feature quality.

    Parameters:
    - target_h: Target edge homophily (0 to 1)
    - feature_separability: How distinguishable features are (0=same, 1=very different)
    """
    set_seed(seed)

    # Generate class labels (balanced)
    labels = np.array([i % n_classes for i in range(n_nodes)])
    np.random.shuffle(labels)

    # Generate features with controlled separability
    # Low separability = classes have similar features (harder to classify)
    # High separability = classes have distinct features (easier to classify)
    features = np.random.randn(n_nodes, n_features)

    # Add class-dependent signal
    class_centers = {}
    for c in range(n_classes):
        # Class centers are spread based on separability
        class_centers[c] = np.random.randn(n_features) * feature_separability * 2

    for i in range(n_nodes):
        features[i] += class_centers[labels[i]]

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Generate edges with target homophily
    n_edges = n_nodes * 15 // 2  # Average degree ~15 (undirected)

    edge_list = []
    same_class_edges = 0
    diff_class_edges = 0

    # Calculate how many same-class vs different-class edges we need
    target_same = int(n_edges * target_h)
    target_diff = n_edges - target_same

    # Generate same-class edges
    for _ in range(target_same * 3):  # Oversample then trim
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

    # Convert to edge_index format
    edge_index = np.array(edge_list).T
    # Make undirected
    edge_index = np.hstack([edge_index, edge_index[[1, 0]]])

    # Calculate actual homophily
    actual_h = same_class_edges / (same_class_edges + diff_class_edges) if (same_class_edges + diff_class_edges) > 0 else 0

    return {
        'features': torch.tensor(features, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'h_target': target_h,
        'h_actual': actual_h,
        'n_nodes': n_nodes,
        'n_edges': edge_index.shape[1],
        'feature_separability': feature_separability
    }

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

def run_separability_sweep():
    """Run experiments across different feature separabilities and homophily values."""

    # Experiment parameters
    separabilities = [0.3, 0.5, 0.7, 1.0]  # Low, Medium, High, Very High
    h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_runs = 5

    results = defaultdict(list)

    print("="*70)
    print("Feature Separability Sweep Experiment")
    print("Testing U-shape robustness across different feature qualities")
    print("="*70)

    for sep in separabilities:
        print(f"\n{'='*60}")
        print(f"Feature Separability: {sep}")
        print(f"{'='*60}")

        for h in h_values:
            gcn_accs = []
            mlp_accs = []

            for run in range(n_runs):
                seed = 42 + run * 100 + int(h * 10) + int(sep * 100)
                data = generate_controlled_graph(
                    target_h=h,
                    feature_separability=sep,
                    seed=seed
                )

                # Train GCN
                gcn = GCN(20, 64, 2)
                gcn_acc = train_and_evaluate(gcn, data)
                gcn_accs.append(gcn_acc)

                # Train MLP
                mlp = MLP(20, 64, 2)
                mlp_acc = train_and_evaluate(mlp, data)
                mlp_accs.append(mlp_acc)

            gcn_mean = np.mean(gcn_accs)
            mlp_mean = np.mean(mlp_accs)
            advantage = gcn_mean - mlp_mean

            results[sep].append({
                'h': h,
                'gcn_acc': gcn_mean,
                'mlp_acc': mlp_mean,
                'gcn_advantage': advantage,
                'gcn_std': np.std(gcn_accs),
                'mlp_std': np.std(mlp_accs)
            })

            winner = "GCN" if advantage > 0 else "MLP"
            print(f"  h={h:.1f}: GCN={gcn_mean:.3f}, MLP={mlp_mean:.3f}, Adv={advantage:+.3f} [{winner}]")

    # Save results
    output = {
        'experiment': 'feature_separability_sweep',
        'separabilities': separabilities,
        'h_values': h_values,
        'n_runs': n_runs,
        'results': {str(sep): results[sep] for sep in separabilities}
    }

    output_path = Path(__file__).parent / "separability_sweep_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print summary analysis
    print("\n" + "="*70)
    print("SUMMARY: U-Shape Consistency Across Feature Separabilities")
    print("="*70)

    for sep in separabilities:
        print(f"\nSeparability={sep}:")
        low_h_wins = sum(1 for r in results[sep] if r['h'] < 0.3 and r['gcn_advantage'] > 0)
        mid_h_wins = sum(1 for r in results[sep] if 0.3 <= r['h'] <= 0.7 and r['gcn_advantage'] > 0)
        high_h_wins = sum(1 for r in results[sep] if r['h'] > 0.7 and r['gcn_advantage'] > 0)

        low_h_total = sum(1 for r in results[sep] if r['h'] < 0.3)
        mid_h_total = sum(1 for r in results[sep] if 0.3 <= r['h'] <= 0.7)
        high_h_total = sum(1 for r in results[sep] if r['h'] > 0.7)

        print(f"  Low h (<0.3): GCN wins {low_h_wins}/{low_h_total}")
        print(f"  Mid h (0.3-0.7): GCN wins {mid_h_wins}/{mid_h_total}")
        print(f"  High h (>0.7): GCN wins {high_h_wins}/{high_h_total}")

        # Check if U-shape pattern holds
        u_shape = (low_h_wins >= low_h_total // 2) and (mid_h_wins <= mid_h_total // 2) and (high_h_wins >= high_h_total // 2)
        print(f"  U-shape pattern: {'YES' if u_shape else 'NO/PARTIAL'}")

    return output

if __name__ == "__main__":
    results = run_separability_sweep()
