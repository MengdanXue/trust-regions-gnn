"""
Expand Low-Homophily Datasets Experiment
=========================================

P0 Task: Add more low-h datasets to strengthen Q2 quadrant validation.

New datasets to add:
1. Penn94 (from LINKX paper) - h ≈ 0.47
2. Twitch-gamers (from Platonov et al.) - h ≈ 0.55
3. Genius (from LINKX paper) - h ≈ 0.62
4. Film (WebKB-like) - if available
5. ogbn-proteins - h varies by edge type

Using HeterophilousGraphDataset from PyTorch Geometric:
- Roman-empire: h ≈ 0.05
- Amazon-ratings: h ≈ 0.38
- Minesweeper: h ≈ 0.68
- Tolokers: h ≈ 0.59
- Questions: h ≈ 0.84
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import (
    Planetoid, Amazon, WebKB, WikipediaNetwork,
    HeterophilousGraphDataset, Actor, LINKXDataset
)
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def compute_homophily(data):
    """Compute edge homophily"""
    edge_index = to_undirected(data.edge_index)
    src, dst = edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()
    return (labels[src] == labels[dst]).mean()


def compute_2hop_homophily(data):
    """Compute 2-hop homophily for recovery ratio analysis"""
    from torch_geometric.utils import to_dense_adj

    edge_index = to_undirected(data.edge_index)
    n = data.num_nodes
    labels = data.y.cpu().numpy()

    # Build adjacency matrix
    adj = torch.zeros(n, n)
    src, dst = edge_index.cpu()
    adj[src, dst] = 1

    # 2-hop adjacency (exclude 1-hop)
    adj2 = torch.mm(adj, adj)
    adj2 = (adj2 > 0).float() - adj - torch.eye(n)
    adj2 = (adj2 > 0).float()

    # Compute 2-hop homophily
    rows, cols = adj2.nonzero(as_tuple=True)
    if len(rows) == 0:
        return 0.0

    same_label = (labels[rows.numpy()] == labels[cols.numpy()]).mean()
    return float(same_label)


def compute_spi(h):
    """Structural Predictability Index"""
    return abs(2 * h - 1)


def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       lr=0.01, weight_decay=5e-4, epochs=200, patience=50):
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


def evaluate_dataset(data, n_runs=5):
    """Evaluate GCN and MLP on a dataset"""
    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    labels = data.y.to(device)
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    gcn_results = []
    mlp_results = []

    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)

        indices = np.arange(n_nodes)
        train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        gcn = GCN(n_features, 64, n_classes).to(device)
        gcn_acc = train_and_evaluate(gcn, x, edge_index, labels, train_mask, val_mask, test_mask)
        gcn_results.append(gcn_acc)

        mlp = MLP(n_features, 64, n_classes).to(device)
        mlp_acc = train_and_evaluate(mlp, x, edge_index, labels, train_mask, val_mask, test_mask)
        mlp_results.append(mlp_acc)

    return {
        'gcn_mean': np.mean(gcn_results),
        'gcn_std': np.std(gcn_results),
        'mlp_mean': np.mean(mlp_results),
        'mlp_std': np.std(mlp_results),
        'gcn_mlp': np.mean(gcn_results) - np.mean(mlp_results),
    }


def main():
    print("=" * 80)
    print("EXPAND LOW-HOMOPHILY DATASETS EXPERIMENT")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGoal: Add more low-h datasets to strengthen Q2 quadrant validation.\n")

    # Extended dataset configuration
    datasets_config = [
        # Original datasets
        ('Cora', Planetoid, {'name': 'Cora'}),
        ('CiteSeer', Planetoid, {'name': 'CiteSeer'}),
        ('PubMed', Planetoid, {'name': 'PubMed'}),
        ('Computers', Amazon, {'name': 'Computers'}),
        ('Photo', Amazon, {'name': 'Photo'}),
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Cornell', WebKB, {'name': 'Cornell'}),
        ('Squirrel', WikipediaNetwork, {'name': 'Squirrel', 'geom_gcn_preprocess': True}),
        ('Chameleon', WikipediaNetwork, {'name': 'Chameleon', 'geom_gcn_preprocess': True}),
        ('Actor', Actor, {}),

        # HeterophilousGraphDataset (Platonov et al. ICLR 2023)
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
        ('Amazon-ratings', HeterophilousGraphDataset, {'name': 'Amazon-ratings'}),
        ('Minesweeper', HeterophilousGraphDataset, {'name': 'Minesweeper'}),
        ('Tolokers', HeterophilousGraphDataset, {'name': 'Tolokers'}),
        ('Questions', HeterophilousGraphDataset, {'name': 'Questions'}),

        # LINKX datasets (Lim et al. NeurIPS 2021) - NEW
        ('Penn94', LINKXDataset, {'name': 'Penn94'}),
        ('genius', LINKXDataset, {'name': 'genius'}),
        ('twitch-gamers', LINKXDataset, {'name': 'twitch-gamers'}),
    ]

    results = []

    print("Loading datasets and computing metrics...")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Nodes':>8} {'Edges':>10} {'h':>7} {'SPI':>7} {'h2':>7} {'R':>7} {'MLP':>7} {'GCN':>7} {'Diff':>8}")
    print("-" * 80)

    for name, DatasetClass, kwargs in datasets_config:
        try:
            dataset = DatasetClass(root='./data', **kwargs)
            data = dataset[0]

            # Compute metrics
            h = compute_homophily(data)
            spi = compute_spi(h)

            # Compute 2-hop homophily (for small graphs only)
            if data.num_nodes < 10000:
                h2 = compute_2hop_homophily(data)
                recovery_ratio = h2 / h if h > 0 else 0
            else:
                h2 = -1
                recovery_ratio = -1

            # Evaluate models
            eval_result = evaluate_dataset(data, n_runs=5)

            result = {
                'dataset': name,
                'num_nodes': data.num_nodes,
                'num_edges': data.edge_index.size(1) // 2,
                'num_features': data.num_features,
                'num_classes': len(data.y.unique()),
                'homophily': h,
                'spi': spi,
                'h2_hop': h2,
                'recovery_ratio': recovery_ratio,
                'mlp_mean': eval_result['mlp_mean'],
                'mlp_std': eval_result['mlp_std'],
                'gcn_mean': eval_result['gcn_mean'],
                'gcn_std': eval_result['gcn_std'],
                'gcn_mlp': eval_result['gcn_mlp'],
            }
            results.append(result)

            h2_str = f"{h2:.3f}" if h2 >= 0 else "N/A"
            r_str = f"{recovery_ratio:.2f}" if recovery_ratio >= 0 else "N/A"

            print(f"{name:<20} {data.num_nodes:>8} {data.edge_index.size(1)//2:>10} "
                  f"{h:>7.3f} {spi:>7.3f} {h2_str:>7} {r_str:>7} "
                  f"{eval_result['mlp_mean']:>7.3f} {eval_result['gcn_mean']:>7.3f} "
                  f"{eval_result['gcn_mlp']:>+8.3f}")

        except Exception as e:
            print(f"{name:<20} Error: {e}")

    print("-" * 80)
    print(f"\nTotal datasets processed: {len(results)}")

    # Analyze low-h datasets
    print("\n" + "=" * 80)
    print("LOW-HOMOPHILY DATASETS ANALYSIS (h < 0.5)")
    print("=" * 80)

    low_h_datasets = [r for r in results if r['homophily'] < 0.5]
    low_h_datasets.sort(key=lambda x: x['homophily'])

    print(f"\n{'Dataset':<20} {'h':>7} {'SPI':>7} {'h2':>7} {'R':>7} {'MLP':>7} {'GCN-MLP':>9} {'Winner':>8}")
    print("-" * 85)

    for r in low_h_datasets:
        h2_str = f"{r['h2_hop']:.3f}" if r['h2_hop'] >= 0 else "N/A"
        r_str = f"{r['recovery_ratio']:.2f}" if r['recovery_ratio'] >= 0 else "N/A"
        winner = "GCN" if r['gcn_mlp'] > 0.01 else ("MLP" if r['gcn_mlp'] < -0.01 else "Tie")

        print(f"{r['dataset']:<20} {r['homophily']:>7.3f} {r['spi']:>7.3f} "
              f"{h2_str:>7} {r_str:>7} {r['mlp_mean']:>7.3f} "
              f"{r['gcn_mlp']:>+9.3f} {winner:>8}")

    # Q2 Quadrant Analysis (High FS, Low h)
    print("\n" + "=" * 80)
    print("Q2 QUADRANT ANALYSIS (High FS >= 0.65, Low h < 0.5)")
    print("=" * 80)

    q2_datasets = [r for r in results if r['mlp_mean'] >= 0.65 and r['homophily'] < 0.5]

    print(f"\nQ2 datasets found: {len(q2_datasets)}")

    if q2_datasets:
        print(f"\n{'Dataset':<20} {'h':>7} {'MLP':>7} {'GCN-MLP':>9} {'Expected':>10} {'Actual':>8}")
        print("-" * 70)

        correct = 0
        for r in q2_datasets:
            expected = "MLP"  # In Q2, MLP should win
            actual = "GCN" if r['gcn_mlp'] > 0.01 else ("MLP" if r['gcn_mlp'] < -0.01 else "Tie")
            is_correct = actual in ["MLP", "Tie"]
            correct += 1 if is_correct else 0
            status = "Y" if is_correct else "N"

            print(f"{r['dataset']:<20} {r['homophily']:>7.3f} {r['mlp_mean']:>7.3f} "
                  f"{r['gcn_mlp']:>+9.3f} {expected:>10} {actual:>8} [{status}]")

        print(f"\nQ2 Accuracy: {correct}/{len(q2_datasets)} = {correct/len(q2_datasets):.1%}")

    # 2-Hop Recovery Analysis
    print("\n" + "=" * 80)
    print("2-HOP RECOVERY RATIO ANALYSIS")
    print("=" * 80)

    low_h_with_r = [r for r in low_h_datasets if r['recovery_ratio'] >= 0]

    if len(low_h_with_r) >= 3:
        from scipy import stats

        recovery_ratios = [r['recovery_ratio'] for r in low_h_with_r]
        gcn_advantages = [r['gcn_mlp'] for r in low_h_with_r]

        r_corr, p_val = stats.pearsonr(recovery_ratios, gcn_advantages)

        print(f"\nCorrelation between Recovery Ratio and GCN Advantage:")
        print(f"  Pearson r: {r_corr:.3f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  N datasets: {len(low_h_with_r)}")

        # Split by recovery ratio threshold
        high_r = [r for r in low_h_with_r if r['recovery_ratio'] > 1.5]
        low_r = [r for r in low_h_with_r if r['recovery_ratio'] <= 1.5]

        print(f"\nHigh Recovery (R > 1.5): {len(high_r)} datasets")
        for r in high_r:
            print(f"  {r['dataset']}: R={r['recovery_ratio']:.2f}, GCN-MLP={r['gcn_mlp']:+.3f}")

        print(f"\nLow Recovery (R <= 1.5): {len(low_r)} datasets")
        for r in low_r:
            print(f"  {r['dataset']}: R={r['recovery_ratio']:.2f}, GCN-MLP={r['gcn_mlp']:+.3f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'all_results': results,
        'low_h_datasets': low_h_datasets,
        'q2_datasets': q2_datasets,
        'summary': {
            'total_datasets': len(results),
            'low_h_count': len(low_h_datasets),
            'q2_count': len(q2_datasets),
        }
    }

    output_path = 'expanded_lowh_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
