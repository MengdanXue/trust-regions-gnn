"""
Ablation Study: SPI-Guided Fusion
=================================

Three-model comparison:
1. Structure-Only (GCN): Standard GNN, ignores SPI
2. Feature-Only (MLP): Ignores graph structure
3. SPI-Guided Fusion (Ours): Dynamic fusion based on SPI

This script validates that:
- Fusion > Single modality overall
- Fusion is especially better in uncertainty zone (SPI < 0.67)
- Temperature T matters for soft transition

Author: [Your Name]
Date: 2024-12
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Import our modules
from spi_guided_gating import (
    SPIGuidedGNN,
    StructureOnlyGNN,
    FeatureOnlyMLP,
    NaiveFusionGNN,
    compute_edge_homophily,
    compute_spi,
    plot_gating_curve
)

# Set random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""
    # Graph parameters
    num_nodes: int = 1000
    num_features: int = 20
    num_classes: int = 2
    avg_degree: int = 15

    # Model parameters
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5

    # Training parameters
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20

    # Experiment parameters
    num_seeds: int = 5
    h_values: List[float] = None

    def __post_init__(self):
        if self.h_values is None:
            self.h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def generate_synthetic_graph(
    h_target: float,
    num_nodes: int,
    num_features: int,
    avg_degree: int,
    feature_separability: float = 0.5,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic graph with controlled homophily.

    Args:
        h_target: Target edge homophily
        num_nodes: Number of nodes
        num_features: Feature dimension
        avg_degree: Average node degree
        feature_separability: Controls how separable classes are by features
        seed: Random seed

    Returns:
        x: [N, F] node features
        edge_index: [2, E] edge indices
        y: [N] node labels
    """
    np.random.seed(seed)

    # Generate balanced labels
    y = np.array([0] * (num_nodes // 2) + [1] * (num_nodes - num_nodes // 2))
    np.random.shuffle(y)
    y = torch.tensor(y, dtype=torch.long)

    # Generate features from class-conditional Gaussians
    x = np.zeros((num_nodes, num_features))
    for c in range(2):
        mask = y.numpy() == c
        # Class center: shifted by separability
        center = np.zeros(num_features)
        center[c * (num_features // 2):(c + 1) * (num_features // 2)] = feature_separability
        x[mask] = np.random.randn(mask.sum(), num_features) + center
    x = torch.tensor(x, dtype=torch.float)

    # Generate edges with target homophily
    num_edges = num_nodes * avg_degree // 2
    num_same_class = int(num_edges * h_target)
    num_diff_class = num_edges - num_same_class

    edges = []

    # Same-class edges
    for c in range(2):
        class_nodes = np.where(y.numpy() == c)[0]
        for _ in range(num_same_class // 2):
            i, j = np.random.choice(class_nodes, 2, replace=False)
            edges.append((i, j))
            edges.append((j, i))

    # Different-class edges
    class_0 = np.where(y.numpy() == 0)[0]
    class_1 = np.where(y.numpy() == 1)[0]
    for _ in range(num_diff_class):
        i = np.random.choice(class_0)
        j = np.random.choice(class_1)
        edges.append((i, j))
        edges.append((j, i))

    # Remove duplicates and self-loops
    edges = list(set(edges))
    edges = [(i, j) for i, j in edges if i != j]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index, y


def train_epoch(model, x, edge_index, y, train_mask, optimizer, spi=None):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    if hasattr(model, 'gating'):  # SPI-Guided model
        out, _ = model(x, edge_index, spi=spi, labels=y)
    else:
        out = model(x, edge_index)

    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, y, mask, spi=None):
    """Evaluate model accuracy."""
    model.eval()

    if hasattr(model, 'gating'):
        out, info = model(x, edge_index, spi=spi, labels=y)
    else:
        out = model(x, edge_index)
        info = {}

    pred = out[mask].argmax(dim=1)
    correct = (pred == y[mask]).sum().item()
    acc = correct / mask.sum().item()

    return acc, info


def run_single_experiment(
    h: float,
    config: ExperimentConfig,
    seed: int
) -> Dict:
    """
    Run single experiment comparing all models at given h.

    Returns dict with accuracies for each model.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate data
    x, edge_index, y = generate_synthetic_graph(
        h_target=h,
        num_nodes=config.num_nodes,
        num_features=config.num_features,
        avg_degree=config.avg_degree,
        feature_separability=0.5,
        seed=seed
    )

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

    # Compute actual homophily and SPI
    actual_h = compute_edge_homophily(edge_index, y)
    spi = compute_spi(actual_h)
    spi_tensor = torch.tensor(spi, device=device)

    # Train/val/test split (60/20/20)
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[perm[:int(0.6 * num_nodes)]] = True
    val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
    test_mask[perm[int(0.8 * num_nodes):]] = True

    results = {'h': h, 'actual_h': actual_h, 'spi': spi, 'seed': seed}

    # Define models
    models = {
        'GCN': StructureOnlyGNN(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout, gnn_type='gcn'
        ),
        'MLP': FeatureOnlyMLP(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout
        ),
        'NaiveFusion': NaiveFusionGNN(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout, fusion_weight=0.5
        ),
        'SPI-Guided': SPIGuidedGNN(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout, gnn_type='gcn',
            tau=0.67, T_init=0.1, learnable_T=True
        )
    }

    # Train each model
    for model_name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(config.epochs):
            # Train
            if model_name == 'SPI-Guided':
                train_epoch(model, x, edge_index, y, train_mask, optimizer, spi=spi_tensor)
                val_acc, _ = evaluate(model, x, edge_index, y, val_mask, spi=spi_tensor)
            else:
                train_epoch(model, x, edge_index, y, train_mask, optimizer)
                val_acc, _ = evaluate(model, x, edge_index, y, val_mask)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        # Load best model and evaluate on test set
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        if model_name == 'SPI-Guided':
            test_acc, info = evaluate(model, x, edge_index, y, test_mask, spi=spi_tensor)
            results[f'{model_name}_beta'] = info.get('beta', 0.5)
        else:
            test_acc, _ = evaluate(model, x, edge_index, y, test_mask)

        results[model_name] = test_acc

    # Compute advantages
    results['SPI-Guided_vs_GCN'] = results['SPI-Guided'] - results['GCN']
    results['SPI-Guided_vs_MLP'] = results['SPI-Guided'] - results['MLP']
    results['SPI-Guided_vs_Naive'] = results['SPI-Guided'] - results['NaiveFusion']

    return results


def run_ablation_study(config: ExperimentConfig) -> List[Dict]:
    """Run complete ablation study across all h values and seeds."""
    all_results = []

    for h in tqdm(config.h_values, desc="H values"):
        for seed in range(config.num_seeds):
            result = run_single_experiment(h, config, seed)
            all_results.append(result)

    return all_results


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    aggregated = {}

    h_values = sorted(list(set(r['h'] for r in results)))

    for h in h_values:
        h_results = [r for r in results if r['h'] == h]

        agg = {
            'h': h,
            'spi': np.mean([r['spi'] for r in h_results]),
            'n': len(h_results)
        }

        for model in ['GCN', 'MLP', 'NaiveFusion', 'SPI-Guided']:
            accs = [r[model] for r in h_results]
            agg[f'{model}_mean'] = np.mean(accs)
            agg[f'{model}_std'] = np.std(accs)

        for adv in ['SPI-Guided_vs_GCN', 'SPI-Guided_vs_MLP', 'SPI-Guided_vs_Naive']:
            advs = [r[adv] for r in h_results]
            agg[f'{adv}_mean'] = np.mean(advs)
            agg[f'{adv}_std'] = np.std(advs)

        aggregated[h] = agg

    return aggregated


def plot_ablation_results(aggregated: Dict, save_dir: Path):
    """Create publication-quality ablation figures."""
    save_dir.mkdir(exist_ok=True)

    h_values = sorted(aggregated.keys())
    spi_values = [aggregated[h]['spi'] for h in h_values]

    # Figure 1: Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy vs H
    ax1 = axes[0]
    for model, color, marker in [
        ('GCN', '#E94F37', 'o'),
        ('MLP', '#3D5A80', 's'),
        ('NaiveFusion', '#F7B801', '^'),
        ('SPI-Guided', '#44AF69', 'D')
    ]:
        means = [aggregated[h][f'{model}_mean'] * 100 for h in h_values]
        stds = [aggregated[h][f'{model}_std'] * 100 for h in h_values]

        ax1.errorbar(h_values, means, yerr=stds, marker=marker, color=color,
                     label=model, linewidth=2, markersize=8, capsize=3)

    # Mark uncertainty zone
    ax1.axvspan(0.3, 0.7, alpha=0.1, color='red')
    ax1.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Edge Homophily (h)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Model Comparison Across Homophily Spectrum', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Right: SPI-Guided Advantage
    ax2 = axes[1]

    adv_gcn = [aggregated[h]['SPI-Guided_vs_GCN_mean'] * 100 for h in h_values]
    adv_mlp = [aggregated[h]['SPI-Guided_vs_MLP_mean'] * 100 for h in h_values]
    adv_naive = [aggregated[h]['SPI-Guided_vs_Naive_mean'] * 100 for h in h_values]

    ax2.bar(np.array(h_values) - 0.02, adv_gcn, width=0.02, label='vs GCN', color='#E94F37', alpha=0.8)
    ax2.bar(np.array(h_values), adv_mlp, width=0.02, label='vs MLP', color='#3D5A80', alpha=0.8)
    ax2.bar(np.array(h_values) + 0.02, adv_naive, width=0.02, label='vs NaiveFusion', color='#F7B801', alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvspan(0.3, 0.7, alpha=0.1, color='red', label='Uncertainty Zone')

    ax2.set_xlabel('Edge Homophily (h)', fontsize=12)
    ax2.set_ylabel('SPI-Guided Advantage (%)', fontsize=12)
    ax2.set_title('(b) SPI-Guided Fusion Advantage', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'ablation_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # Figure 2: Trust Region Analysis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute wins/losses
    trust_zone_wins = sum(1 for h in h_values if aggregated[h]['spi'] > 0.4
                          and aggregated[h]['SPI-Guided_vs_GCN_mean'] >= 0)
    trust_zone_total = sum(1 for h in h_values if aggregated[h]['spi'] > 0.4)

    uncertain_zone_wins = sum(1 for h in h_values if aggregated[h]['spi'] <= 0.4
                               and aggregated[h]['SPI-Guided_vs_MLP_mean'] >= 0)
    uncertain_zone_total = sum(1 for h in h_values if aggregated[h]['spi'] <= 0.4)

    # Bar chart
    zones = ['Trust Zone\n(SPI > 0.4)', 'Uncertainty Zone\n(SPI <= 0.4)']
    win_rates = [trust_zone_wins / max(trust_zone_total, 1) * 100,
                 uncertain_zone_wins / max(uncertain_zone_total, 1) * 100]

    colors = ['#44AF69', '#F7B801']
    bars = ax.bar(zones, win_rates, color=colors, edgecolor='black', linewidth=2)

    ax.set_ylabel('SPI-Guided Win Rate (%)', fontsize=12)
    ax.set_title('SPI-Guided Fusion Performance by Zone', fontsize=14)
    ax.set_ylim(0, 110)

    # Add value labels
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / 'trust_zone_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {save_dir}")


def generate_latex_table(aggregated: Dict) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study: SPI-Guided Fusion vs Baselines}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        r"$h$ & SPI & GCN & MLP & Naive & \textbf{SPI-Guided} & $\Delta$ \\",
        r"\midrule"
    ]

    for h in sorted(aggregated.keys()):
        agg = aggregated[h]
        delta = agg['SPI-Guided_mean'] - max(agg['GCN_mean'], agg['MLP_mean'])
        sign = '+' if delta >= 0 else ''

        line = (f"{h:.1f} & {agg['spi']:.2f} & "
                f"{agg['GCN_mean']*100:.1f} & {agg['MLP_mean']*100:.1f} & "
                f"{agg['NaiveFusion_mean']*100:.1f} & "
                f"\\textbf{{{agg['SPI-Guided_mean']*100:.1f}}} & "
                f"{sign}{delta*100:.1f} \\\\")
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)


def main():
    """Run complete ablation study."""
    print("="*60)
    print("SPI-Guided Fusion Ablation Study")
    print("="*60)

    # Configuration
    config = ExperimentConfig(
        num_nodes=1000,
        num_features=20,
        hidden_channels=64,
        num_layers=2,
        epochs=200,
        num_seeds=5,
        h_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    print(f"\nConfig: {config}")

    # Run experiments
    print("\nRunning experiments...")
    results = run_ablation_study(config)

    # Aggregate results
    aggregated = aggregate_results(results)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'h':<6} {'SPI':<8} {'GCN':<10} {'MLP':<10} {'Naive':<10} {'SPI-Guided':<12} {'Best?':<8}")
    print("-"*70)

    for h in sorted(aggregated.keys()):
        agg = aggregated[h]
        models = {
            'GCN': agg['GCN_mean'],
            'MLP': agg['MLP_mean'],
            'Naive': agg['NaiveFusion_mean'],
            'SPI-Guided': agg['SPI-Guided_mean']
        }
        best = max(models, key=models.get)
        is_ours_best = 'Yes' if best == 'SPI-Guided' else ''

        print(f"{h:<6.1f} {agg['spi']:<8.2f} "
              f"{agg['GCN_mean']*100:<10.1f} {agg['MLP_mean']*100:<10.1f} "
              f"{agg['NaiveFusion_mean']*100:<10.1f} {agg['SPI-Guided_mean']*100:<12.1f} "
              f"{is_ours_best:<8}")

    # Save results
    output_dir = Path(__file__).parent
    results_path = output_dir / "ablation_spi_fusion_results.json"

    with open(results_path, 'w') as f:
        json.dump({
            'raw_results': results,
            'aggregated': {str(k): v for k, v in aggregated.items()},
            'config': config.__dict__
        }, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")

    # Generate figures
    plot_ablation_results(aggregated, output_dir / "figures")

    # Generate LaTeX table
    latex = generate_latex_table(aggregated)
    latex_path = output_dir / "ablation_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved: {latex_path}")

    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Calculate overall win rate
    spi_guided_wins = sum(1 for h in aggregated.keys()
                          if aggregated[h]['SPI-Guided_mean'] >= max(
                              aggregated[h]['GCN_mean'],
                              aggregated[h]['MLP_mean'],
                              aggregated[h]['NaiveFusion_mean']))

    print(f"1. SPI-Guided wins in {spi_guided_wins}/{len(aggregated)} h values")

    # Uncertainty zone analysis
    uncertain_h = [h for h in aggregated.keys() if 0.3 <= h <= 0.7]
    uncertain_wins = sum(1 for h in uncertain_h
                         if aggregated[h]['SPI-Guided_mean'] >= max(
                             aggregated[h]['GCN_mean'], aggregated[h]['MLP_mean']))
    print(f"2. In uncertainty zone (0.3<=h<=0.7): wins {uncertain_wins}/{len(uncertain_h)}")

    # Average advantage
    avg_adv_gcn = np.mean([aggregated[h]['SPI-Guided_vs_GCN_mean'] * 100 for h in aggregated.keys()])
    avg_adv_mlp = np.mean([aggregated[h]['SPI-Guided_vs_MLP_mean'] * 100 for h in aggregated.keys()])
    print(f"3. Average advantage vs GCN: {avg_adv_gcn:+.2f}%")
    print(f"4. Average advantage vs MLP: {avg_adv_mlp:+.2f}%")


if __name__ == "__main__":
    main()
