"""
Experiment 1: Two-Hop Homophily Analysis
Purpose: Verify if 2-hop homophily recovers in low-h datasets

Key hypothesis: In heterophilic graphs, 2-hop neighbors may have higher
same-class probability than 1-hop neighbors, explaining why standard GCN fails.
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import PyG
try:
    from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
    from torch_geometric.utils import to_scipy_sparse_matrix
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available, using synthetic data only")

from scipy.sparse import csr_matrix
import scipy.sparse as sp


def compute_1hop_homophily(edge_index, labels):
    """Compute standard 1-hop edge homophily."""
    if isinstance(edge_index, torch.Tensor):
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    else:
        src, dst = edge_index[0], edge_index[1]

    same_label = (labels[src] == labels[dst])
    return same_label.mean()


def compute_2hop_homophily(edge_index, labels, num_nodes):
    """
    Compute 2-hop homophily: probability that a 2-hop neighbor has the same label.

    For each node, we look at neighbors-of-neighbors (excluding the node itself)
    and compute the fraction that share the same label.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # Build adjacency matrix
    src, dst = edge_index[0], edge_index[1]
    data = np.ones(len(src))
    A = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))

    # Make symmetric if not already
    A = A + A.T
    A.data = np.ones_like(A.data)  # Binary adjacency

    # Compute A^2 for 2-hop connections
    A2 = A @ A

    # Remove self-loops and 1-hop connections
    A2.setdiag(0)
    A2 = A2 - A  # Remove 1-hop connections
    A2.data = np.clip(A2.data, 0, 1)  # Binary
    A2.eliminate_zeros()

    if A2.nnz == 0:
        return None  # No valid 2-hop connections

    # Compute 2-hop homophily
    rows, cols = A2.nonzero()
    same_label_2hop = (labels[rows] == labels[cols])

    return same_label_2hop.mean()


def compute_2hop_homophily_sampled(edge_index, labels, num_nodes, sample_size=10000):
    """
    Compute 2-hop homophily with sampling for large graphs.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # Build adjacency list
    adj_list = defaultdict(set)
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        adj_list[s].add(d)
        adj_list[d].add(s)

    # Sample nodes
    nodes = np.random.choice(num_nodes, min(sample_size, num_nodes), replace=False)

    same_count = 0
    total_count = 0

    for node in nodes:
        node_label = labels[node]
        neighbors_1hop = adj_list[node]

        # Get 2-hop neighbors (exclude node itself and 1-hop neighbors)
        neighbors_2hop = set()
        for n1 in neighbors_1hop:
            for n2 in adj_list[n1]:
                if n2 != node and n2 not in neighbors_1hop:
                    neighbors_2hop.add(n2)

        # Count same-label 2-hop neighbors
        for n2 in neighbors_2hop:
            if labels[n2] == node_label:
                same_count += 1
            total_count += 1

    if total_count == 0:
        return None

    return same_count / total_count


def load_real_datasets():
    """Load real-world datasets for analysis."""
    datasets = {}
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    if not HAS_PYG:
        return datasets

    # Homophilic datasets
    try:
        for name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'edge_index': data.edge_index,
                'labels': data.y,
                'num_nodes': data.num_nodes,
                'type': 'homophilic'
            }
    except Exception as e:
        print(f"Could not load Planetoid datasets: {e}")

    # Heterophilic datasets (WebKB)
    try:
        for name in ['Texas', 'Wisconsin', 'Cornell']:
            dataset = WebKB(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'edge_index': data.edge_index,
                'labels': data.y,
                'num_nodes': data.num_nodes,
                'type': 'heterophilic'
            }
    except Exception as e:
        print(f"Could not load WebKB datasets: {e}")

    # Actor dataset
    try:
        dataset = Actor(root=str(data_dir))
        data = dataset[0]
        datasets['actor'] = {
            'edge_index': data.edge_index,
            'labels': data.y,
            'num_nodes': data.num_nodes,
            'type': 'heterophilic'
        }
    except Exception as e:
        print(f"Could not load Actor dataset: {e}")

    # Wikipedia networks
    try:
        for name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name] = {
                'edge_index': data.edge_index,
                'labels': data.y,
                'num_nodes': data.num_nodes,
                'type': 'heterophilic'
            }
    except Exception as e:
        print(f"Could not load Wikipedia datasets: {e}")

    return datasets


def generate_synthetic_graph(num_nodes=1000, h_target=0.5, num_edges_per_node=10, num_classes=2):
    """Generate a synthetic graph with controlled homophily."""
    labels = np.random.randint(0, num_classes, num_nodes)

    edges_src = []
    edges_dst = []

    for i in range(num_nodes):
        for _ in range(num_edges_per_node):
            # Decide if same-class or different-class neighbor
            if np.random.random() < h_target:
                # Same class neighbor
                same_class = np.where(labels == labels[i])[0]
                if len(same_class) > 1:
                    j = np.random.choice(same_class)
                    while j == i:
                        j = np.random.choice(same_class)
                else:
                    j = np.random.randint(0, num_nodes)
            else:
                # Different class neighbor
                diff_class = np.where(labels != labels[i])[0]
                if len(diff_class) > 0:
                    j = np.random.choice(diff_class)
                else:
                    j = np.random.randint(0, num_nodes)

            edges_src.append(i)
            edges_dst.append(j)

    edge_index = np.array([edges_src, edges_dst])
    return edge_index, labels, num_nodes


def run_analysis():
    """Run the 2-hop homophily analysis."""
    print("=" * 60)
    print("Experiment 1: Two-Hop Homophily Analysis")
    print("=" * 60)

    results = {
        'experiment': 'two_hop_homophily_analysis',
        'hypothesis': '2-hop homophily recovers in heterophilic graphs',
        'datasets': []
    }

    # Load real datasets
    print("\nLoading real-world datasets...")
    datasets = load_real_datasets()

    # Add synthetic datasets with varying homophily
    print("Generating synthetic datasets...")
    for h_target in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        edge_index, labels, num_nodes = generate_synthetic_graph(
            num_nodes=2000, h_target=h_target, num_edges_per_node=15
        )
        datasets[f'synthetic_h{h_target:.1f}'] = {
            'edge_index': edge_index,
            'labels': labels,
            'num_nodes': num_nodes,
            'type': 'synthetic',
            'target_h': h_target
        }

    # Analyze each dataset
    print("\nAnalyzing datasets...")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Type':<12} {'1-hop h':<10} {'2-hop h':<10} {'Recovery':<10} {'Ratio':<10}")
    print("-" * 80)

    for name, data in datasets.items():
        edge_index = data['edge_index']
        labels = data['labels']
        num_nodes = data['num_nodes']

        # Compute 1-hop homophily
        h1 = compute_1hop_homophily(edge_index, labels)

        # Compute 2-hop homophily
        if num_nodes > 10000:
            h2 = compute_2hop_homophily_sampled(edge_index, labels, num_nodes)
        else:
            h2 = compute_2hop_homophily(edge_index, labels, num_nodes)

        if h2 is None:
            h2 = h1  # Fallback

        # Compute recovery ratio
        # For heterophilic graphs (h1 < 0.5), we expect h2 > h1
        # Recovery = (h2 - h1) / (1 - h1) if h1 < 0.5
        if h1 < 0.5:
            recovery = (h2 - h1) / max(0.5 - h1, 0.01)  # How much closer to 0.5
        else:
            recovery = 0  # Not applicable for homophilic graphs

        ratio = h2 / max(h1, 0.01)

        # Determine if 2-hop helps
        is_heterophilic = h1 < 0.5
        two_hop_helps = h2 > h1 if is_heterophilic else False

        result = {
            'dataset': name,
            'type': data['type'],
            'num_nodes': num_nodes,
            'homophily_1hop': float(h1),
            'homophily_2hop': float(h2),
            'recovery_ratio': float(recovery),
            'h2_h1_ratio': float(ratio),
            'is_heterophilic': str(is_heterophilic),
            'two_hop_helps': str(two_hop_helps)
        }
        results['datasets'].append(result)

        recovery_str = f"{recovery:.2f}" if is_heterophilic else "N/A"
        print(f"{name:<20} {data['type']:<12} {h1:<10.3f} {h2:<10.3f} {recovery_str:<10} {ratio:<10.2f}")

    print("-" * 80)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary: Does 2-hop Homophily Recover in Heterophilic Graphs?")
    print("=" * 60)

    heterophilic = [d for d in results['datasets'] if d['is_heterophilic']]
    homophilic = [d for d in results['datasets'] if not d['is_heterophilic']]

    if heterophilic:
        avg_h1_hetero = np.mean([d['homophily_1hop'] for d in heterophilic])
        avg_h2_hetero = np.mean([d['homophily_2hop'] for d in heterophilic])
        recovery_count = sum(1 for d in heterophilic if d['two_hop_helps'])

        print(f"\nHeterophilic graphs (n={len(heterophilic)}):")
        print(f"  Average 1-hop homophily: {avg_h1_hetero:.3f}")
        print(f"  Average 2-hop homophily: {avg_h2_hetero:.3f}")
        print(f"  2-hop recovery rate: {recovery_count}/{len(heterophilic)} ({100*recovery_count/len(heterophilic):.1f}%)")
        print(f"  Average improvement: {avg_h2_hetero - avg_h1_hetero:+.3f}")

        results['summary'] = {
            'heterophilic_count': len(heterophilic),
            'avg_1hop_h_heterophilic': float(avg_h1_hetero),
            'avg_2hop_h_heterophilic': float(avg_h2_hetero),
            'recovery_rate': recovery_count / len(heterophilic),
            'avg_improvement': float(avg_h2_hetero - avg_h1_hetero)
        }

    if homophilic:
        avg_h1_homo = np.mean([d['homophily_1hop'] for d in homophilic])
        avg_h2_homo = np.mean([d['homophily_2hop'] for d in homophilic])

        print(f"\nHomophilic graphs (n={len(homophilic)}):")
        print(f"  Average 1-hop homophily: {avg_h1_homo:.3f}")
        print(f"  Average 2-hop homophily: {avg_h2_homo:.3f}")
        print(f"  Change: {avg_h2_homo - avg_h1_homo:+.3f}")

        results['summary']['homophilic_count'] = len(homophilic)
        results['summary']['avg_1hop_h_homophilic'] = float(avg_h1_homo)
        results['summary']['avg_2hop_h_homophilic'] = float(avg_h2_homo)

    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)

    if heterophilic and avg_h2_hetero > avg_h1_hetero:
        print("CONFIRMED: 2-hop homophily DOES recover in heterophilic graphs!")
        print(f"  This supports the dual-channel SPI approach.")
        print(f"  On average, 2-hop homophily is {avg_h2_hetero - avg_h1_hetero:.3f} higher than 1-hop.")
        results['conclusion'] = 'CONFIRMED: 2-hop homophily recovers'
        results['supports_dual_channel'] = 'True'
    else:
        print("NOT CONFIRMED: 2-hop homophily does NOT consistently recover.")
        print("  Alternative explanations for SPI failure should be investigated.")
        results['conclusion'] = 'NOT CONFIRMED: 2-hop does not consistently help'
        results['supports_dual_channel'] = 'False'

    # Save results
    output_path = Path(__file__).parent / 'two_hop_homophily_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_analysis()
