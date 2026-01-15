"""
Process Downloaded Datasets and Run Experiments
================================================

Process manually downloaded Texas, Chameleon, and Roman-empire datasets.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import GCNConv
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


def load_geom_gcn_dataset(node_file, edge_file):
    """Load dataset from geom-gcn format (node_feature_label.txt + graph_edges.txt)"""
    # Load node features and labels
    with open(node_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    node_ids = []
    features_list = []
    labels = []

    for line in lines:
        parts = line.strip().split('\t')
        node_id = int(parts[0])
        feat = list(map(float, parts[1].split(',')))
        label = int(parts[2])

        node_ids.append(node_id)
        features_list.append(feat)
        labels.append(label)

    # Create node id mapping
    id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}

    x = torch.tensor(features_list, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Load edges
    with open(edge_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    edges = []
    for line in lines:
        parts = line.strip().split('\t')
        src, dst = int(parts[0]), int(parts[1])
        if src in id_map and dst in id_map:
            edges.append([id_map[src], id_map[dst]])
            edges.append([id_map[dst], id_map[src]])  # Undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index, y


def load_npz_dataset(npz_file):
    """Load dataset from npz format (HeterophilousGraphDataset)"""
    data = np.load(npz_file)

    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)

    edges = data['edges']
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    return x, edge_index, y


def compute_homophily(edge_index, labels):
    """Compute edge homophily"""
    src, dst = edge_index.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return (labels_np[src] == labels_np[dst]).mean()


def compute_2hop_homophily(edge_index, labels, n_nodes):
    """Compute 2-hop homophily"""
    adj = torch.zeros(n_nodes, n_nodes)
    src, dst = edge_index.cpu()
    adj[src, dst] = 1

    # 2-hop adjacency (exclude 1-hop and self)
    adj2 = torch.mm(adj, adj)
    adj2 = (adj2 > 0).float() - adj - torch.eye(n_nodes)
    adj2 = (adj2 > 0).float()

    rows, cols = adj2.nonzero(as_tuple=True)
    if len(rows) == 0:
        return 0.0

    labels_np = labels.cpu().numpy()
    same_label = (labels_np[rows.numpy()] == labels_np[cols.numpy()]).mean()
    return float(same_label)


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


def evaluate_dataset(x, edge_index, labels, n_runs=5):
    """Evaluate GCN and MLP on dataset"""
    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    n_nodes = x.size(0)
    n_features = x.size(1)
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
    print("PROCESS DOWNLOADED DATASETS")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = './data'
    results = []

    # Process Texas
    print("\n" + "-" * 40)
    print("Processing Texas...")
    try:
        x, edge_index, y = load_geom_gcn_dataset(
            f'{data_dir}/texas_node_feature_label.txt',
            f'{data_dir}/texas_graph_edges.txt'
        )
        h = compute_homophily(edge_index, y)
        h2 = compute_2hop_homophily(edge_index, y, x.size(0))
        recovery_ratio = h2 / h if h > 0 else 0

        eval_result = evaluate_dataset(x, edge_index, y, n_runs=5)

        result = {
            'dataset': 'Texas',
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1) // 2,
            'homophily': h,
            'h2_hop': h2,
            'recovery_ratio': recovery_ratio,
            'mlp_mean': eval_result['mlp_mean'],
            'gcn_mean': eval_result['gcn_mean'],
            'gcn_mlp': eval_result['gcn_mlp'],
        }
        results.append(result)
        print(f"  Nodes: {x.size(0)}, Edges: {edge_index.size(1)//2}")
        print(f"  h={h:.3f}, h2={h2:.3f}, R={recovery_ratio:.2f}")
        print(f"  MLP={eval_result['mlp_mean']:.3f}, GCN={eval_result['gcn_mean']:.3f}, Diff={eval_result['gcn_mlp']:+.3f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Process Chameleon
    print("\n" + "-" * 40)
    print("Processing Chameleon...")
    try:
        x, edge_index, y = load_geom_gcn_dataset(
            f'{data_dir}/chameleon_node_feature_label.txt',
            f'{data_dir}/chameleon_graph_edges.txt'
        )
        h = compute_homophily(edge_index, y)
        h2 = compute_2hop_homophily(edge_index, y, x.size(0))
        recovery_ratio = h2 / h if h > 0 else 0

        eval_result = evaluate_dataset(x, edge_index, y, n_runs=5)

        result = {
            'dataset': 'Chameleon',
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1) // 2,
            'homophily': h,
            'h2_hop': h2,
            'recovery_ratio': recovery_ratio,
            'mlp_mean': eval_result['mlp_mean'],
            'gcn_mean': eval_result['gcn_mean'],
            'gcn_mlp': eval_result['gcn_mlp'],
        }
        results.append(result)
        print(f"  Nodes: {x.size(0)}, Edges: {edge_index.size(1)//2}")
        print(f"  h={h:.3f}, h2={h2:.3f}, R={recovery_ratio:.2f}")
        print(f"  MLP={eval_result['mlp_mean']:.3f}, GCN={eval_result['gcn_mean']:.3f}, Diff={eval_result['gcn_mlp']:+.3f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Process Roman-empire
    print("\n" + "-" * 40)
    print("Processing Roman-empire...")
    try:
        x, edge_index, y = load_npz_dataset(f'{data_dir}/roman_empire.npz')
        h = compute_homophily(edge_index, y)
        # Skip 2-hop for large graph
        h2 = -1
        recovery_ratio = -1

        eval_result = evaluate_dataset(x, edge_index, y, n_runs=5)

        result = {
            'dataset': 'Roman-empire',
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1) // 2,
            'homophily': h,
            'h2_hop': h2,
            'recovery_ratio': recovery_ratio,
            'mlp_mean': eval_result['mlp_mean'],
            'gcn_mean': eval_result['gcn_mean'],
            'gcn_mlp': eval_result['gcn_mlp'],
        }
        results.append(result)
        print(f"  Nodes: {x.size(0)}, Edges: {edge_index.size(1)//2}")
        print(f"  h={h:.3f}")
        print(f"  MLP={eval_result['mlp_mean']:.3f}, GCN={eval_result['gcn_mean']:.3f}, Diff={eval_result['gcn_mlp']:+.3f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':<15} {'h':>7} {'h2':>7} {'R':>7} {'MLP':>7} {'GCN':>7} {'Diff':>8} {'Winner':>8}")
    print("-" * 75)

    for r in results:
        h2_str = f"{r['h2_hop']:.3f}" if r['h2_hop'] >= 0 else "N/A"
        r_str = f"{r['recovery_ratio']:.2f}" if r['recovery_ratio'] >= 0 else "N/A"
        winner = "GCN" if r['gcn_mlp'] > 0.01 else ("MLP" if r['gcn_mlp'] < -0.01 else "Tie")

        print(f"{r['dataset']:<15} {r['homophily']:>7.3f} {h2_str:>7} {r_str:>7} "
              f"{r['mlp_mean']:>7.3f} {r['gcn_mean']:>7.3f} {r['gcn_mlp']:>+8.3f} {winner:>8}")

    # Q2 Analysis
    print("\n" + "-" * 40)
    print("Q2 QUADRANT (High FS >= 0.65, Low h < 0.5):")
    q2 = [r for r in results if r['mlp_mean'] >= 0.65 and r['homophily'] < 0.5]
    for r in q2:
        winner = "MLP" if r['gcn_mlp'] < -0.01 else ("GCN" if r['gcn_mlp'] > 0.01 else "Tie")
        expected = "MLP"
        correct = "Y" if winner in ["MLP", "Tie"] else "N"
        print(f"  {r['dataset']}: h={r['homophily']:.3f}, MLP={r['mlp_mean']:.3f}, "
              f"Diff={r['gcn_mlp']:+.3f} -> {winner} (Expected: {expected}) [{correct}]")

    if q2:
        q2_correct = sum(1 for r in q2 if r['gcn_mlp'] <= 0.01)
        print(f"\n  Q2 Accuracy: {q2_correct}/{len(q2)} = {q2_correct/len(q2):.0%}")

    # Save results
    output_path = 'downloaded_datasets_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
