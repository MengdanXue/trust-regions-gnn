"""
OGB Dataset Validation
======================

验证Trust Regions Framework在OGB大规模数据集上的表现。
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json

# Fix PyTorch 2.6 weights_only issue
import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr])
except:
    pass
try:
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([GlobalStorage])
except:
    pass

from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def compute_homophily(edge_index, labels):
    """Compute edge homophily"""
    src, dst = edge_index.cpu().numpy()
    lab = labels.cpu().numpy().squeeze()
    valid = (lab[src] >= 0) & (lab[dst] >= 0)  # Filter out unknown labels
    if valid.sum() == 0:
        return 0.5
    return (lab[src][valid] == lab[dst][valid]).mean()


def train_ogb(model, data, optimizer, train_idx):
    """Train one epoch"""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_ogb(model, data, split_idx, evaluator):
    """Evaluate on all splits"""
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        results[split] = evaluator.eval({
            'y_true': data.y[idx],
            'y_pred': y_pred[idx],
        })['acc']

    return results


def run_ogb_experiment(dataset_name, n_runs=3, epochs=200):
    """Run experiment on OGB dataset"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = PygNodePropPredDataset(name=dataset_name, root='./data')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=dataset_name)

    # Move to device
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    # Compute homophily
    h = compute_homophily(data.edge_index, data.y)

    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Homophily: {h:.4f}")

    mlp_scores = []
    gcn_scores = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}:")

        # MLP
        mlp = MLP(data.num_features, 256, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

        best_val = 0
        best_test = 0
        for epoch in range(epochs):
            train_ogb(mlp, data, optimizer, train_idx)
            if (epoch + 1) % 50 == 0:
                res = evaluate_ogb(mlp, data, split_idx, evaluator)
                if res['valid'] > best_val:
                    best_val = res['valid']
                    best_test = res['test']

        mlp_scores.append(best_test)
        print(f"    MLP: {best_test:.4f}")

        # GCN
        gcn = GCN(data.num_features, 256, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)

        best_val = 0
        best_test = 0
        for epoch in range(epochs):
            train_ogb(gcn, data, optimizer, train_idx)
            if (epoch + 1) % 50 == 0:
                res = evaluate_ogb(gcn, data, split_idx, evaluator)
                if res['valid'] > best_val:
                    best_val = res['valid']
                    best_test = res['test']

        gcn_scores.append(best_test)
        print(f"    GCN: {best_test:.4f}")

    # Summary
    mlp_mean = np.mean(mlp_scores)
    gcn_mean = np.mean(gcn_scores)
    delta = gcn_mean - mlp_mean

    print(f"\n  Summary:")
    print(f"    MLP: {mlp_mean:.4f} +/- {np.std(mlp_scores):.4f}")
    print(f"    GCN: {gcn_mean:.4f} +/- {np.std(gcn_scores):.4f}")
    print(f"    Delta: {delta:+.4f}")

    # Classify quadrant
    fs_thresh, h_thresh = 0.65, 0.5
    if mlp_mean >= fs_thresh:
        if h >= h_thresh:
            quadrant, prediction = 'Q1', 'GCN_maybe'
        else:
            quadrant, prediction = 'Q2', 'MLP'
    else:
        if h >= h_thresh:
            quadrant, prediction = 'Q3', 'GCN'
        else:
            quadrant, prediction = 'Q4', 'Uncertain'

    winner = 'GCN' if delta > 0.01 else ('MLP' if delta < -0.01 else 'Tie')

    print(f"\n  Quadrant: {quadrant}")
    print(f"  Prediction: {prediction}")
    print(f"  Actual: {winner}")

    return {
        'name': dataset_name,
        'n_nodes': data.num_nodes,
        'n_edges': data.num_edges,
        'homophily': h,
        'mlp_mean': mlp_mean,
        'gcn_mean': gcn_mean,
        'delta': delta,
        'quadrant': quadrant,
        'prediction': prediction,
        'winner': winner
    }


def main():
    print("=" * 80)
    print("OGB LARGE-SCALE DATASET VALIDATION")
    print("=" * 80)

    # Only run ogbn-arxiv (manageable size)
    results = []

    try:
        result = run_ogb_experiment('ogbn-arxiv', n_runs=3, epochs=200)
        results.append(result)
    except Exception as e:
        print(f"Failed: {e}")

    # Save results
    if results:
        with open('ogb_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to: ogb_validation_results.json")

    return results


if __name__ == '__main__':
    results = main()
