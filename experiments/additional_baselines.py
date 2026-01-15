"""
Additional Heterophily-Aware Baselines
======================================

MixHop, GCNII, GAT 完整对比实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict

from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset, Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, degree
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# Model Implementations
# ============================================================

class MLP(nn.Module):
    """Baseline MLP"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GCN(nn.Module):
    """Standard GCN"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# ============================================================
# MixHop Implementation (Fixed)
# ============================================================

class MixHop(nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures (ICML 2019)

    Fixed version with correct dimension handling.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Layer 1: Mix 0-hop, 1-hop, 2-hop
        # Each hop produces hidden_channels // 3 features
        hop_dim = hidden_channels // 3
        self.lin0_1 = nn.Linear(in_channels, hop_dim)
        self.lin1_1 = nn.Linear(in_channels, hop_dim)
        self.lin2_1 = nn.Linear(in_channels, hop_dim)

        # Actual hidden dim after concat
        actual_hidden = hop_dim * 3

        # Layer 2: Output layer
        self.lin_out = nn.Linear(actual_hidden, out_channels)

    def forward(self, x, edge_index):
        # Compute normalized adjacency
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 0-hop (identity)
        h0 = self.lin0_1(x)

        # 1-hop (A @ x)
        ax = self._propagate(x, edge_index, norm)
        h1 = self.lin1_1(ax)

        # 2-hop (A^2 @ x)
        a2x = self._propagate(ax, edge_index, norm)
        h2 = self.lin2_1(a2x)

        # Concatenate
        h = torch.cat([h0, h1, h2], dim=-1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.lin_out(h)

    def _propagate(self, x, edge_index, norm):
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))
        return out


# ============================================================
# GCNII Implementation
# ============================================================

class GCNIIConv(nn.Module):
    """GCNII layer with initial residual and identity mapping"""
    def __init__(self, hidden_channels, alpha=0.1, theta=0.5, layer=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.theta = theta
        self.layer = layer
        self.lin = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, x, x0, edge_index, norm):
        # Propagation
        row, col = edge_index
        h = torch.zeros_like(x)
        h.index_add_(0, col, x[row] * norm.view(-1, 1))

        # Initial residual connection
        h = (1 - self.alpha) * h + self.alpha * x0

        # Identity mapping
        beta = np.log(self.theta / self.layer + 1)
        h = (1 - beta) * h + beta * self.lin(h)

        return h


class GCNII(nn.Module):
    """
    GCNII: Simple and Deep Graph Convolutional Networks (ICML 2020)

    Key innovations:
    1. Initial residual connection
    2. Identity mapping
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=8, alpha=0.1, theta=0.5, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNIIConv(hidden_channels, alpha, theta, layer=i+1))

    def forward(self, x, edge_index):
        # Compute normalization
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Initial projection
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x0 = x  # Save initial representation

        # GCNII layers
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index, norm))

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       lr=0.01, weight_decay=5e-4, epochs=200, patience=50):
    """Train model and return best test accuracy"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_test_acc


def compute_homophily(edge_index, labels):
    """Compute edge homophily"""
    src, dst = edge_index.cpu().numpy()
    lab = labels.cpu().numpy()
    return (lab[src] == lab[dst]).mean()


# ============================================================
# Main Experiment
# ============================================================

def run_additional_baselines(n_runs=10):
    """
    Run MixHop, GCNII, GAT comparison.
    """

    print("=" * 80)
    print("ADDITIONAL BASELINES: MixHop, GCNII, GAT")
    print("=" * 80)

    # Datasets
    datasets_config = [
        # Q2 Quadrant
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Cornell', WebKB, {'name': 'Cornell'}),
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
        # Trust Region
        ('Cora', Planetoid, {'name': 'Cora'}),
        ('CiteSeer', Planetoid, {'name': 'CiteSeer'}),
        ('PubMed', Planetoid, {'name': 'PubMed'}),
    ]

    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'GAT': GAT,
        'MixHop': MixHop,
        'GCNII': GCNII,
    }

    all_results = {}

    for ds_name, DatasetClass, kwargs in datasets_config:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            # Load dataset
            dataset = DatasetClass(root='./data', **kwargs)
            data = dataset[0]

            x = data.x.to(device)
            edge_index = to_undirected(data.edge_index).to(device)
            labels = data.y.to(device)
            n_nodes = data.num_nodes
            n_features = data.num_features
            n_classes = len(labels.unique())

            h = compute_homophily(edge_index, labels)
            spi = abs(2 * h - 1)

            print(f"  Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
            print(f"  Homophily: {h:.4f}, SPI: {spi:.4f}")

            # Determine regime
            if h < 0.3:
                regime = "Q2 (Low h)"
            elif h > 0.7:
                regime = "Trust Region (High h)"
            else:
                regime = "Uncertain"
            print(f"  Regime: {regime}")

            results = {model_name: [] for model_name in model_classes}

            for seed in range(n_runs):
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create splits
                indices = np.arange(n_nodes)
                train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=seed)
                val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed)

                train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                train_mask[train_idx] = True
                val_mask[val_idx] = True
                test_mask[test_idx] = True

                # Train each model
                for model_name, ModelClass in model_classes.items():
                    try:
                        if model_name == 'GAT':
                            # Ensure hidden_channels is divisible by heads
                            model = ModelClass(n_features, 64, n_classes, heads=8).to(device)
                        elif model_name == 'GCNII':
                            model = ModelClass(n_features, 64, n_classes, num_layers=8).to(device)
                        else:
                            model = ModelClass(n_features, 64, n_classes).to(device)

                        acc = train_and_evaluate(model, x, edge_index, labels,
                                                train_mask, val_mask, test_mask)
                        results[model_name].append(acc)
                    except Exception as e:
                        print(f"    Error with {model_name} seed {seed}: {e}")
                        results[model_name].append(0.0)

            # Print results
            print(f"\n  Results ({n_runs} runs):")
            print(f"  {'Model':>12} {'Mean':>10} {'Std':>10}")
            print("  " + "-" * 35)

            for model_name in model_classes:
                scores = results[model_name]
                if scores and any(s > 0 for s in scores):
                    mean = np.mean([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
                    std = np.std([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
                    print(f"  {model_name:>12} {mean*100:>10.1f}% {std*100:>10.1f}%")
                else:
                    print(f"  {model_name:>12} {'FAILED':>10}")

            all_results[ds_name] = {
                'homophily': h,
                'spi': spi,
                'regime': regime,
                'n_nodes': n_nodes,
                'results': {k: {'mean': np.mean(v) if v else 0, 'std': np.std(v) if v else 0, 'scores': v}
                           for k, v in results.items()}
            }

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n| Dataset | h | Regime | MLP | GCN | GAT | MixHop | GCNII | Best |")
    print("|---------|-----|--------|-----|-----|-----|--------|-------|------|")

    for ds_name, data in all_results.items():
        h = data['homophily']
        regime = data['regime'][:8]

        scores = {}
        for m in ['MLP', 'GCN', 'GAT', 'MixHop', 'GCNII']:
            if m in data['results'] and data['results'][m]['mean'] > 0:
                scores[m] = data['results'][m]['mean']
            else:
                scores[m] = 0.0

        best = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else 'N/A'

        print(f"| {ds_name:13} | {h:.2f} | {regime:8} | {scores['MLP']*100:.1f} | {scores['GCN']*100:.1f} | {scores['GAT']*100:.1f} | {scores['MixHop']*100:.1f} | {scores['GCNII']*100:.1f} | {best} |")

    # Save results
    results_file = 'additional_baselines_results.json'
    json_results = {}
    for ds_name, data in all_results.items():
        json_results[ds_name] = {
            'homophily': float(data['homophily']),
            'spi': float(data['spi']),
            'regime': data['regime'],
            'n_nodes': int(data['n_nodes']),
            'results': {
                k: {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'scores': [float(s) for s in v['scores']]
                }
                for k, v in data['results'].items()
            }
        }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == '__main__':
    results = run_additional_baselines(n_runs=10)
