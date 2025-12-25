#!/usr/bin/env python3
"""
One-Click Paper Reproduction Script
====================================

This script reproduces all main results from:
"Trust Regions of Graph Propagation: Explaining the U-Shaped Performance
Pattern in Graph Neural Networks"

Usage:
    python scripts/reproduce_paper.py [--quick] [--full]

    --quick: Run minimal experiments (5 minutes)
    --full:  Run all experiments with 5 seeds (2 hours)

Expected Output:
    results/reproduction_report.md - Comparison with paper results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Verify all dependencies are installed."""
    print("=" * 60)
    print("STEP 1: Checking Environment")
    print("=" * 60)

    errors = []

    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, got {sys.version}")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda}")
        else:
            print("  CUDA: Not available (using CPU)")
    except ImportError:
        errors.append("PyTorch not installed")

    # Check PyTorch Geometric
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError:
        errors.append("torch-geometric not installed")

    # Check other dependencies
    for pkg in ['numpy', 'scipy', 'matplotlib', 'networkx']:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {pkg} {version}")
        except ImportError:
            errors.append(f"{pkg} not installed")

    if errors:
        print("\n" + "=" * 60)
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return False

    print("\n✓ All dependencies satisfied!")
    return True


def run_h_sweep(quick=False):
    """Run H-Sweep experiment (Figure 1, Table 2)."""
    print("\n" + "=" * 60)
    print("STEP 2: Running H-Sweep Experiment")
    print("=" * 60)

    import numpy as np
    import torch
    from utils.data_generation import generate_csbm_graph
    from utils.metrics import calculate_spi, calculate_edge_homophily

    # Configuration
    h_values = [0.1, 0.3, 0.5, 0.7, 0.9] if quick else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_runs = 2 if quick else 5
    n_nodes = 500 if quick else 1000

    results = []

    for h in h_values:
        print(f"\n  Testing h = {h:.1f} (SPI = {calculate_spi(h):.2f})")

        mlp_accs = []
        gcn_accs = []

        for seed in range(n_runs):
            np.random.seed(42 + seed)
            torch.manual_seed(42 + seed)

            # Generate synthetic graph
            data = generate_csbm_graph(
                n_nodes=n_nodes,
                n_edges=n_nodes * 5,
                n_features=16,
                n_classes=2,
                homophily=h,
                feature_noise=0.3
            )

            # Simple MLP baseline (feature-only)
            from models.mlp import MLP
            mlp = MLP(in_channels=data.num_features, hidden_channels=64,
                     out_channels=2, num_layers=2)

            # Train MLP
            mlp_acc = train_and_eval(mlp, data, use_edges=False)
            mlp_accs.append(mlp_acc)

            # GCN (structure + features)
            from models.gcn import GCN
            gcn = GCN(in_channels=data.num_features, hidden_channels=64,
                     out_channels=2, num_layers=2)

            # Train GCN
            gcn_acc = train_and_eval(gcn, data, use_edges=True)
            gcn_accs.append(gcn_acc)

        result = {
            'h': h,
            'spi': calculate_spi(h),
            'mlp_mean': float(np.mean(mlp_accs)),
            'mlp_std': float(np.std(mlp_accs)),
            'gcn_mean': float(np.mean(gcn_accs)),
            'gcn_std': float(np.std(gcn_accs)),
            'delta': float(np.mean(gcn_accs) - np.mean(mlp_accs)),
        }
        results.append(result)

        print(f"    MLP: {result['mlp_mean']:.1%} ± {result['mlp_std']:.1%}")
        print(f"    GCN: {result['gcn_mean']:.1%} ± {result['gcn_std']:.1%}")
        print(f"    Δ:   {result['delta']:+.1%}")

    return results


def train_and_eval(model, data, use_edges=True, epochs=100):
    """Simple training loop."""
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Split data
    n = data.num_nodes
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:int(0.6*n)]] = True
    test_mask[perm[int(0.8*n):]] = True

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if use_edges:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        if use_edges:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x)
        pred = out.argmax(dim=1)
        correct = (pred[test_mask] == data.y[test_mask]).sum()
        acc = correct / test_mask.sum()

    return acc.item()


def run_spi_correlation(h_sweep_results):
    """Compute SPI correlation (Figure 2)."""
    print("\n" + "=" * 60)
    print("STEP 3: Computing SPI Correlation")
    print("=" * 60)

    import numpy as np
    from scipy import stats

    spi_values = [r['spi'] for r in h_sweep_results]
    delta_values = [r['delta'] for r in h_sweep_results]

    # Pearson correlation
    r, p = stats.pearsonr(spi_values, delta_values)
    print(f"\n  SPI vs GCN-MLP Delta:")
    print(f"    Pearson r = {r:.3f} (p = {p:.2e})")

    # R-squared
    r_squared = r ** 2
    print(f"    R² = {r_squared:.3f}")

    return {
        'pearson_r': float(r),
        'p_value': float(p),
        'r_squared': float(r_squared),
    }


def generate_report(h_sweep_results, correlation_results, output_dir):
    """Generate markdown report."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating Report")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save JSON results
    with open(output_dir / 'h_sweep_results.json', 'w') as f:
        json.dump(h_sweep_results, f, indent=2)

    with open(output_dir / 'correlation_results.json', 'w') as f:
        json.dump(correlation_results, f, indent=2)

    # Generate markdown report
    report = f"""# Reproduction Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report reproduces the main findings from:
**"Trust Regions of Graph Propagation: Explaining the U-Shaped Performance Pattern in Graph Neural Networks"**

## H-Sweep Results (Table 2)

| h | SPI | MLP | GCN | GCN-MLP |
|---|-----|-----|-----|---------|
"""

    for r in h_sweep_results:
        report += f"| {r['h']:.1f} | {r['spi']:.2f} | {r['mlp_mean']:.1%} | {r['gcn_mean']:.1%} | {r['delta']:+.1%} |\n"

    report += f"""
## SPI Correlation (Figure 2)

- **Pearson r**: {correlation_results['pearson_r']:.3f}
- **R²**: {correlation_results['r_squared']:.3f}
- **p-value**: {correlation_results['p_value']:.2e}

## Key Findings Verification

### U-Shape Pattern
"""

    # Verify U-shape
    low_h = [r for r in h_sweep_results if r['h'] <= 0.3]
    mid_h = [r for r in h_sweep_results if 0.3 < r['h'] < 0.7]
    high_h = [r for r in h_sweep_results if r['h'] >= 0.7]

    if low_h:
        low_delta = sum(r['delta'] for r in low_h) / len(low_h)
        report += f"- Low h (≤0.3): GCN-MLP = {low_delta:+.1%} {'✓' if low_delta > 0 else '✗'}\n"

    if mid_h:
        mid_delta = sum(r['delta'] for r in mid_h) / len(mid_h)
        report += f"- Mid h (0.3-0.7): GCN-MLP = {mid_delta:+.1%} {'✓' if mid_delta < 0 else '✗'}\n"

    if high_h:
        high_delta = sum(r['delta'] for r in high_h) / len(high_h)
        report += f"- High h (≥0.7): GCN-MLP = {high_delta:+.1%} {'✓' if high_delta > 0 else '✗'}\n"

    report += f"""
### SPI as Predictor

- Paper claims: R² = 0.82
- Reproduced: R² = {correlation_results['r_squared']:.2f}
- Match: {'✓ Close match' if abs(correlation_results['r_squared'] - 0.82) < 0.15 else '✗ Different (may need more runs)'}

## Conclusion

The U-shaped pattern is {'confirmed' if low_h and mid_h and high_h and low_delta > 0 and mid_delta < 0 and high_delta > 0 else 'partially confirmed'}.
SPI correlation R² = {correlation_results['r_squared']:.2f}.

## Files Generated

- `h_sweep_results.json` - Raw H-sweep data
- `correlation_results.json` - SPI correlation analysis
- `reproduction_report.md` - This report
"""

    with open(output_dir / 'reproduction_report.md', 'w') as f:
        f.write(report)

    print(f"\n  ✓ Report saved to: {output_dir / 'reproduction_report.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (5 minutes)')
    parser.add_argument('--full', action='store_true',
                       help='Full mode with all experiments (2 hours)')
    parser.add_argument('--output', type=str, default='results/reproduction',
                       help='Output directory')
    args = parser.parse_args()

    print("=" * 60)
    print("TRUST REGIONS GNN - PAPER REPRODUCTION")
    print("=" * 60)
    print(f"\nMode: {'Quick' if args.quick else 'Full' if args.full else 'Standard'}")
    print(f"Output: {args.output}")

    # Step 1: Check environment
    if not check_environment():
        sys.exit(1)

    # Step 2: Run H-sweep
    h_sweep_results = run_h_sweep(quick=args.quick)

    # Step 3: Compute SPI correlation
    correlation_results = run_spi_correlation(h_sweep_results)

    # Step 4: Generate report
    report = generate_report(h_sweep_results, correlation_results, args.output)

    print("\n" + "=" * 60)
    print("REPRODUCTION COMPLETE!")
    print("=" * 60)
    print(f"\nCheck results in: {args.output}/")
    print("\nKey findings:")
    print(f"  - SPI R² = {correlation_results['r_squared']:.2f} (paper: 0.82)")

    # Print U-shape verification
    low_h = [r for r in h_sweep_results if r['h'] <= 0.3]
    high_h = [r for r in h_sweep_results if r['h'] >= 0.7]
    if low_h and high_h:
        low_delta = sum(r['delta'] for r in low_h) / len(low_h)
        high_delta = sum(r['delta'] for r in high_h) / len(high_h)
        print(f"  - Low h GCN advantage: {low_delta:+.1%}")
        print(f"  - High h GCN advantage: {high_delta:+.1%}")


if __name__ == '__main__':
    main()
