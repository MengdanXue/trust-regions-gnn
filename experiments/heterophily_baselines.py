"""
Heterophily-Aware Baselines Comparison
======================================

补充实验：FAGCN, LINKX, MixHop 与 MLP/GCN 对比
验证：即使使用专门设计的heterophily方法，Q2象限仍然是MLP占优

References:
- FAGCN: Beyond Low-frequency Information in Graph Convolutional Networks (AAAI 2021)
- LINKX: Large Scale Learning on Non-Homophilous Graphs (NeurIPS 2021)
- MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing (ICML 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset, Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, degree, get_laplacian
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# Model Implementations
# ============================================================

class MLP(nn.Module):
    """Baseline MLP (no graph structure)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(nn.Module):
    """Standard GCN"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ============================================================
# FAGCN Implementation
# ============================================================

class FAGCNConv(MessagePassing):
    """FAGCN: Frequency Adaptive Graph Convolution"""
    def __init__(self, in_channels, eps=0.1):
        super().__init__(aggr='add')
        self.eps = eps
        self.att = nn.Parameter(torch.Tensor(1, in_channels))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Low-pass and high-pass
        x_l = self.propagate(edge_index, x=x, norm=norm, mode='low')
        x_h = self.propagate(edge_index, x=x, norm=norm, mode='high')

        # Adaptive combination
        gate = torch.sigmoid((x * self.att).sum(dim=-1, keepdim=True))
        return gate * x_l + (1 - gate) * x_h

    def message(self, x_j, norm, mode):
        if mode == 'low':
            return norm.view(-1, 1) * x_j
        else:
            return -self.eps * norm.view(-1, 1) * x_j


class FAGCN(nn.Module):
    """FAGCN: Beyond Low-frequency Information in GCN (AAAI 2021)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, eps=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(FAGCNConv(hidden_channels, eps))
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
        return self.lin2(x)


# ============================================================
# LINKX Implementation
# ============================================================

class LINKX(nn.Module):
    """LINKX: Large Scale Learning on Non-Homophilous Graphs (NeurIPS 2021)

    Separates feature and structure processing, then combines.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Feature MLP
        self.mlp_feat = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Structure MLP (processes A*X)
        self.mlp_struct = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Final MLP
        self.mlp_final = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # Feature branch
        h_feat = self.mlp_feat(x)

        # Structure branch: simple message passing
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0

        # A * X (normalized)
        ax = torch.zeros_like(x)
        ax.index_add_(0, col, x[row] * deg_inv[row].view(-1, 1))
        h_struct = self.mlp_struct(ax)

        # Combine
        h = torch.cat([h_feat, h_struct], dim=-1)
        return self.mlp_final(h)


# ============================================================
# MixHop Implementation
# ============================================================

class MixHopConv(MessagePassing):
    """MixHop: Multi-hop neighborhood mixing"""
    def __init__(self, in_channels, out_channels, hops=[0, 1, 2]):
        super().__init__(aggr='add')
        self.hops = hops
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels // len(hops)) for _ in hops
        ])

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        outputs = []
        x_hop = x
        for i, hop in enumerate(self.hops):
            if hop == 0:
                outputs.append(self.lins[i](x))
            else:
                for _ in range(hop):
                    x_hop = self.propagate(edge_index, x=x_hop, norm=norm)
                outputs.append(self.lins[i](x_hop))
                x_hop = x  # Reset for next hop computation

        return torch.cat(outputs, dim=-1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class MixHop(nn.Module):
    """MixHop: Higher-Order Graph Convolutional Architectures (ICML 2019)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(MixHopConv(in_channels, hidden_channels, hops=[0, 1, 2]))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(MixHopConv(hidden_channels, hidden_channels, hops=[0, 1, 2]))
        # Final layer
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


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

def run_heterophily_baselines(n_runs=10):
    """
    Compare heterophily-aware methods with MLP and GCN.
    """

    print("=" * 80)
    print("HETEROPHILY-AWARE BASELINES COMPARISON")
    print("=" * 80)
    print("\nMethods: MLP, GCN, FAGCN, LINKX, MixHop")
    print("Question: Can heterophily-aware methods beat MLP in Q2 quadrant?\n")

    # Datasets
    datasets_config = [
        # Q2 Quadrant: High FS, Low h
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Cornell', WebKB, {'name': 'Cornell'}),
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
        # Trust Region: High h
        ('Cora', Planetoid, {'name': 'Cora'}),
        ('CiteSeer', Planetoid, {'name': 'CiteSeer'}),
        ('PubMed', Planetoid, {'name': 'PubMed'}),
    ]

    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'FAGCN': FAGCN,
        'LINKX': LINKX,
        'MixHop': MixHop,
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
                        model = ModelClass(n_features, 64, n_classes).to(device)
                        acc = train_and_evaluate(model, x, edge_index, labels,
                                                train_mask, val_mask, test_mask)
                        results[model_name].append(acc)
                    except Exception as e:
                        print(f"    Error with {model_name}: {e}")
                        results[model_name].append(0.0)

            # Print results
            print(f"\n  Results ({n_runs} runs):")
            print(f"  {'Model':>12} {'Mean':>10} {'Std':>10}")
            print("  " + "-" * 35)

            for model_name in model_classes:
                scores = results[model_name]
                if scores:
                    mean = np.mean(scores)
                    std = np.std(scores)
                    print(f"  {model_name:>12} {mean*100:>10.1f}% {std*100:>10.1f}%")

            all_results[ds_name] = {
                'homophily': h,
                'spi': spi,
                'regime': regime,
                'n_nodes': n_nodes,
                'results': {k: {'mean': np.mean(v), 'std': np.std(v), 'scores': v}
                           for k, v in results.items() if v}
            }

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ============================================================
    # Summary Analysis
    # ============================================================

    print("\n" + "=" * 80)
    print("SUMMARY: Heterophily-Aware Methods vs SPI Predictions")
    print("=" * 80)

    print("\n| Dataset | h | Regime | MLP | GCN | FAGCN | LINKX | MixHop | Best |")
    print("|---------|-----|--------|-----|-----|-------|-------|--------|------|")

    for ds_name, data in all_results.items():
        h = data['homophily']
        regime = data['regime'][:8]

        scores = {}
        for m in ['MLP', 'GCN', 'FAGCN', 'LINKX', 'MixHop']:
            if m in data['results']:
                scores[m] = data['results'][m]['mean']
            else:
                scores[m] = 0.0

        best = max(scores, key=scores.get)

        print(f"| {ds_name:13} | {h:.2f} | {regime:8} | {scores['MLP']*100:.1f} | {scores['GCN']*100:.1f} | {scores['FAGCN']*100:.1f} | {scores['LINKX']*100:.1f} | {scores['MixHop']*100:.1f} | {best} |")

    # Q2 Analysis
    print("\n" + "-" * 60)
    print("Q2 Quadrant Analysis (High FS, Low h):")
    print("-" * 60)

    q2_datasets = ['Texas', 'Wisconsin', 'Cornell', 'Roman-empire']
    q2_wins = defaultdict(int)

    for ds_name in q2_datasets:
        if ds_name in all_results:
            data = all_results[ds_name]
            scores = {m: data['results'].get(m, {}).get('mean', 0) for m in model_classes}
            winner = max(scores, key=scores.get)
            q2_wins[winner] += 1
            print(f"  {ds_name}: Best = {winner} ({scores[winner]*100:.1f}%)")

    print(f"\nQ2 Quadrant Winners:")
    for model, wins in sorted(q2_wins.items(), key=lambda x: -x[1]):
        print(f"  {model}: {wins}/4")

    # Save results
    results_file = 'heterophily_baselines_results.json'
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
    results = run_heterophily_baselines(n_runs=10)
