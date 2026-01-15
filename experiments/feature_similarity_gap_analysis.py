"""
Experiment 3: Feature Similarity Gap Analysis
Purpose: Analyze if heterophilic neighbors are "opposite" (contrastive) or "orthogonal" (noise)

Key hypothesis: Real heterophilic neighbors are feature-orthogonal (noise),
not feature-opposite (contrastive signal). This explains why GCN fails on low-h graphs.
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available")

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def compute_feature_similarity_gap(features, edge_index, labels):
    """
    Compute the feature similarity gap between same-class and different-class neighbors.

    Returns:
        intra_sim: Average cosine similarity between same-class neighbors
        inter_sim: Average cosine similarity between different-class neighbors
        gap: intra_sim - inter_sim (positive = good for classification)
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    src, dst = edge_index[0], edge_index[1]

    # Sample edges if too many
    if len(src) > 50000:
        idx = np.random.choice(len(src), 50000, replace=False)
        src, dst = src[idx], dst[idx]

    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # Compute similarities for each edge
    sims = np.sum(features_norm[src] * features_norm[dst], axis=1)

    # Separate by same/different class
    same_class_mask = labels[src] == labels[dst]
    diff_class_mask = ~same_class_mask

    intra_sim = sims[same_class_mask].mean() if same_class_mask.sum() > 0 else 0
    inter_sim = sims[diff_class_mask].mean() if diff_class_mask.sum() > 0 else 0

    return float(intra_sim), float(inter_sim), float(intra_sim - inter_sim)


def compute_opposite_vs_orthogonal(features, edge_index, labels):
    """
    Determine if different-class neighbors are:
    - Opposite (negative similarity): contrastive signal, GNN can exploit
    - Orthogonal (near-zero similarity): noise, GNN cannot exploit
    - Similar (positive similarity): confusion, GNN will fail
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    src, dst = edge_index[0], edge_index[1]

    # Get different-class edges
    diff_class_mask = labels[src] != labels[dst]
    src_diff = src[diff_class_mask]
    dst_diff = dst[diff_class_mask]

    if len(src_diff) == 0:
        return 'N/A', 0, {}

    # Sample if too many
    if len(src_diff) > 50000:
        idx = np.random.choice(len(src_diff), 50000, replace=False)
        src_diff, dst_diff = src_diff[idx], dst_diff[idx]

    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # Compute similarities for different-class edges
    sims = np.sum(features_norm[src_diff] * features_norm[dst_diff], axis=1)

    # Categorize
    opposite_frac = (sims < -0.1).mean()  # Negative similarity
    orthogonal_frac = (np.abs(sims) <= 0.1).mean()  # Near zero
    similar_frac = (sims > 0.1).mean()  # Positive similarity

    # Determine dominant type
    if opposite_frac > 0.4:
        dominant = 'opposite'
    elif orthogonal_frac > 0.4:
        dominant = 'orthogonal'
    else:
        dominant = 'similar'

    stats = {
        'opposite_frac': float(opposite_frac),
        'orthogonal_frac': float(orthogonal_frac),
        'similar_frac': float(similar_frac),
        'mean_sim': float(sims.mean()),
        'std_sim': float(sims.std())
    }

    return dominant, float(sims.mean()), stats


def load_datasets():
    """Load real-world datasets."""
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
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
                'type': 'homophilic'
            }
    except Exception as e:
        print(f"Could not load Planetoid: {e}")

    # Heterophilic datasets
    try:
        for name in ['Texas', 'Wisconsin', 'Cornell']:
            dataset = WebKB(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
                'type': 'heterophilic'
            }
    except Exception as e:
        print(f"Could not load WebKB: {e}")

    try:
        dataset = Actor(root=str(data_dir))
        data = dataset[0]
        datasets['actor'] = {
            'features': data.x,
            'edge_index': data.edge_index,
            'labels': data.y,
            'type': 'heterophilic'
        }
    except Exception as e:
        print(f"Could not load Actor: {e}")

    try:
        for name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
                'type': 'heterophilic'
            }
    except Exception as e:
        print(f"Could not load Wikipedia: {e}")

    return datasets


def compute_homophily(edge_index, labels):
    """Compute edge homophily."""
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    src, dst = edge_index[0], edge_index[1]
    return (labels[src] == labels[dst]).mean()


def run_analysis():
    """Run feature similarity gap analysis."""
    print("=" * 70)
    print("Experiment 3: Feature Similarity Gap Analysis")
    print("=" * 70)
    print("\nHypothesis: Real heterophilic neighbors are feature-ORTHOGONAL (noise),")
    print("            not feature-OPPOSITE (contrastive signal).")
    print("            This explains why GCN fails on low-h real graphs.\n")

    results = {
        'experiment': 'feature_similarity_gap_analysis',
        'hypothesis': 'heterophilic neighbors are orthogonal not opposite',
        'datasets': []
    }

    datasets = load_datasets()

    if not datasets:
        print("No datasets available")
        return None

    print("-" * 100)
    print(f"{'Dataset':<12} {'Type':<12} {'h':<8} {'Intra':<8} {'Inter':<8} {'Gap':<8} {'Dominant':<12} {'Exploitable?':<12}")
    print("-" * 100)

    for name, data in datasets.items():
        features = data['features']
        edge_index = data['edge_index']
        labels = data['labels']

        # Compute homophily
        h = compute_homophily(edge_index, labels)

        # Compute feature similarity gap
        intra_sim, inter_sim, gap = compute_feature_similarity_gap(features, edge_index, labels)

        # Analyze if opposite or orthogonal
        dominant, mean_inter_sim, stats = compute_opposite_vs_orthogonal(features, edge_index, labels)

        # Determine if GNN can exploit structure
        # GNN can exploit if:
        # 1. High h AND positive gap (homophilic + features align)
        # 2. Low h AND negative inter_sim (heterophilic + features opposite)
        is_heterophilic = h < 0.5
        if is_heterophilic:
            exploitable = dominant == 'opposite'  # Can exploit negative correlation
        else:
            exploitable = gap > 0.05  # Can exploit positive correlation

        result = {
            'dataset': name,
            'type': data['type'],
            'homophily': float(h),
            'intra_class_sim': intra_sim,
            'inter_class_sim': inter_sim,
            'similarity_gap': gap,
            'neighbor_type': dominant,
            'neighbor_stats': stats,
            'structure_exploitable': str(exploitable)
        }
        results['datasets'].append(result)

        exploit_str = 'Yes' if exploitable else 'No'
        print(f"{name:<12} {data['type']:<12} {h:<8.3f} {intra_sim:<8.3f} {inter_sim:<8.3f} "
              f"{gap:<8.3f} {dominant:<12} {exploit_str:<12}")

    print("-" * 100)

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Feature-Structure Alignment")
    print("=" * 70)

    heterophilic = [d for d in results['datasets'] if d['homophily'] < 0.5]
    homophilic = [d for d in results['datasets'] if d['homophily'] >= 0.5]

    if heterophilic:
        print(f"\nHeterophilic graphs (h < 0.5):")
        opposite_count = sum(1 for d in heterophilic if d['neighbor_type'] == 'opposite')
        orthogonal_count = sum(1 for d in heterophilic if d['neighbor_type'] == 'orthogonal')
        similar_count = sum(1 for d in heterophilic if d['neighbor_type'] == 'similar')

        print(f"  Opposite (exploitable):    {opposite_count}/{len(heterophilic)}")
        print(f"  Orthogonal (noise):        {orthogonal_count}/{len(heterophilic)}")
        print(f"  Similar (confusing):       {similar_count}/{len(heterophilic)}")

        avg_inter_sim = np.mean([d['inter_class_sim'] for d in heterophilic])
        print(f"  Average inter-class sim:   {avg_inter_sim:.3f}")

        results['heterophilic_summary'] = {
            'count': len(heterophilic),
            'opposite': opposite_count,
            'orthogonal': orthogonal_count,
            'similar': similar_count,
            'avg_inter_sim': float(avg_inter_sim)
        }

    if homophilic:
        print(f"\nHomophilic graphs (h >= 0.5):")
        avg_gap = np.mean([d['similarity_gap'] for d in homophilic])
        print(f"  Average similarity gap:    {avg_gap:.3f}")

        results['homophilic_summary'] = {
            'count': len(homophilic),
            'avg_gap': float(avg_gap)
        }

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING:")
    print("=" * 70)

    if heterophilic:
        orthogonal_frac = orthogonal_count / len(heterophilic)
        if orthogonal_frac >= 0.5:
            print("CONFIRMED: Most heterophilic graphs have ORTHOGONAL neighbors (noise)!")
            print("  This explains why GCN fails: aggregation adds noise, not contrastive signal.")
            print("  SPI correctly identifies high theoretical information, but GCN cannot extract it.")
            results['conclusion'] = 'CONFIRMED: orthogonal neighbors explain GCN failure'
        else:
            print("MIXED: Heterophilic graphs show varied neighbor patterns.")
            print("  Some are opposite (exploitable), some are orthogonal (noise).")
            results['conclusion'] = 'MIXED: varied neighbor patterns'

    # Correlation analysis
    print("\n" + "-" * 50)
    print("Correlation: Homophily vs Feature Similarity Gap")

    h_values = [d['homophily'] for d in results['datasets']]
    gap_values = [d['similarity_gap'] for d in results['datasets']]

    if len(h_values) > 2:
        correlation = np.corrcoef(h_values, gap_values)[0, 1]
        print(f"  Pearson r = {correlation:.3f}")

        # This is the key finding from the paper
        if correlation < -0.5:
            print("  NEGATIVE correlation: higher h -> smaller gap")
            print("  This is COUNTER-INTUITIVE and matches the paper's finding (r=-0.755)")
        elif correlation > 0.5:
            print("  POSITIVE correlation: higher h -> larger gap")
            print("  Features and structure are aligned")
        else:
            print("  WEAK correlation: features and structure are independent")

        results['correlation'] = float(correlation)

    # Save results
    output_path = Path(__file__).parent / 'feature_similarity_gap_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_analysis()
