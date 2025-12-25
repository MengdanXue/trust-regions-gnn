"""
SPI vs Delta(GNN-MLP) Correlation Analysis
P0-3: Prove SPI can predict when GNN beats MLP

This script analyzes the correlation between Structural Predictability Index (SPI)
and GNN advantage over MLP across different homophily levels.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

def calculate_spi(h):
    """
    Calculate Structural Predictability Index.
    SPI = |2h - 1|

    Interpretation:
    - h=0.5 → SPI=0 (no structural signal)
    - h=0 or h=1 → SPI=1 (maximum structural signal)
    """
    return abs(2 * h - 1)

def load_all_data():
    """Load data from multiple experiments."""
    base_path = Path(__file__).parent
    all_data = []

    # 1. Cross-model H-sweep data
    cross_model_path = base_path / "cross_model_hsweep_results.json"
    if cross_model_path.exists():
        with open(cross_model_path, 'r') as f:
            data = json.load(f)
        for r in data['results']:
            all_data.append({
                'h': r['h'],
                'spi': calculate_spi(r['h']),
                'gcn_advantage': r['GCN_advantage'] * 100,
                'gat_advantage': r['GAT_advantage'] * 100,
                'sage_advantage': r['GraphSAGE_advantage'] * 100,
                'source': 'cross_model',
                'feature_sep': 0.5
            })

    # 2. Separability sweep data
    sep_path = base_path / "separability_sweep_results.json"
    if sep_path.exists():
        with open(sep_path, 'r') as f:
            sep_data = json.load(f)
        # Results is a dict with separability as key
        for sep_key, h_results in sep_data['results'].items():
            sep = float(sep_key)
            for hr in h_results:
                all_data.append({
                    'h': hr['h'],
                    'spi': calculate_spi(hr['h']),
                    'gcn_advantage': hr['gcn_advantage'] * 100,
                    'gat_advantage': None,
                    'sage_advantage': None,
                    'source': 'separability_sweep',
                    'feature_sep': sep
                })

    return all_data

def analyze_spi_correlation():
    """Analyze correlation between SPI and GNN advantage."""
    data = load_all_data()

    # Extract arrays
    spi_values = np.array([d['spi'] for d in data])
    gcn_adv = np.array([d['gcn_advantage'] for d in data])

    # Pearson correlation
    r, p_value = stats.pearsonr(spi_values, gcn_adv)

    # Spearman correlation (rank-based, more robust)
    rho, p_spearman = stats.spearmanr(spi_values, gcn_adv)

    # Linear regression
    slope, intercept, r_value, p_reg, std_err = stats.linregress(spi_values, gcn_adv)

    print("="*60)
    print("SPI vs GCN Advantage Correlation Analysis")
    print("="*60)
    print(f"N = {len(data)} data points")
    print(f"\nPearson r = {r:.4f} (p = {p_value:.2e})")
    print(f"Spearman rho = {rho:.4f} (p = {p_spearman:.2e})")
    print(f"\nLinear regression: GCN_adv = {slope:.2f} x SPI + {intercept:.2f}")
    print(f"R^2 = {r_value**2:.4f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if r > 0.5:
        print("[+] STRONG positive correlation: Higher SPI -> Higher GCN advantage")
    elif r > 0.3:
        print("[~] MODERATE positive correlation")
    elif r > 0:
        print("[.] WEAK positive correlation")
    else:
        print("[-] No or negative correlation")

    if p_value < 0.001:
        print("[+] Highly statistically significant (p < 0.001)")
    elif p_value < 0.05:
        print("[+] Statistically significant (p < 0.05)")
    else:
        print("[-] Not statistically significant")

    return {
        'pearson_r': r,
        'pearson_p': p_value,
        'spearman_rho': rho,
        'spearman_p': p_spearman,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'n_samples': len(data)
    }

def plot_spi_correlation():
    """Create publication-quality SPI correlation plot."""
    data = load_all_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Left: SPI vs GCN Advantage =====
    ax1 = axes[0]

    # Separate by source
    cross_model = [d for d in data if d['source'] == 'cross_model']
    sep_sweep = [d for d in data if d['source'] == 'separability_sweep']

    # Plot cross-model data
    spi_cm = [d['spi'] for d in cross_model]
    adv_cm = [d['gcn_advantage'] for d in cross_model]
    ax1.scatter(spi_cm, adv_cm, c='#2E86AB', s=100, alpha=0.8,
                label='Cross-Model H-Sweep', marker='o', edgecolors='white', linewidth=1)

    # Plot separability sweep data with color by feature_sep
    seps = sorted(list(set([d['feature_sep'] for d in sep_sweep])))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(seps)))
    for sep, color in zip(seps, colors):
        sep_data = [d for d in sep_sweep if d['feature_sep'] == sep]
        spi_s = [d['spi'] for d in sep_data]
        adv_s = [d['gcn_advantage'] for d in sep_data]
        ax1.scatter(spi_s, adv_s, c=[color], s=60, alpha=0.6,
                   label=f'Sep={sep}', marker='s')

    # Linear regression line
    all_spi = np.array([d['spi'] for d in data])
    all_adv = np.array([d['gcn_advantage'] for d in data])
    slope, intercept, r_value, _, _ = stats.linregress(all_spi, all_adv)
    x_line = np.linspace(0, 1, 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r--', linewidth=2, label=f'Fit: R²={r_value**2:.3f}')

    # Reference line at y=0
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Zone annotation
    ax1.axvspan(0, 0.4, alpha=0.1, color='red')
    ax1.axvspan(0.4, 1.0, alpha=0.1, color='green')
    ax1.text(0.2, -15, 'Low SPI\n(GNN risky)', ha='center', fontsize=10, color='darkred')
    ax1.text(0.7, 5, 'High SPI\n(GNN safe)', ha='center', fontsize=10, color='darkgreen')

    ax1.set_xlabel('Structural Predictability Index (SPI)')
    ax1.set_ylabel('GCN Advantage over MLP (%)')
    ax1.set_title('(a) SPI Predicts GCN Advantage')
    ax1.set_xlim(-0.05, 1.05)
    ax1.legend(loc='lower right', fontsize=9)

    # ===== Right: Trust Region Visualization =====
    ax2 = axes[1]

    # Plot h vs GCN advantage with trust region coloring
    h_values = np.array([d['h'] for d in cross_model])
    gcn_adv = np.array([d['gcn_advantage'] for d in cross_model])

    # Color by zone
    colors_zone = []
    for h in h_values:
        if h < 0.3 or h > 0.7:
            colors_zone.append('#44AF69')  # Green - trust zone
        else:
            colors_zone.append('#E94F37')  # Red - uncertainty zone

    ax2.scatter(h_values, gcn_adv, c=colors_zone, s=150, alpha=0.8,
                edgecolors='white', linewidth=2)

    # Connect with line
    sorted_indices = np.argsort(h_values)
    ax2.plot(h_values[sorted_indices], gcn_adv[sorted_indices],
             'k-', linewidth=1.5, alpha=0.5)

    # Zone shading
    ax2.axvspan(0, 0.3, alpha=0.15, color='green', label='Trust Region (h<0.3 or h>0.7)')
    ax2.axvspan(0.3, 0.7, alpha=0.15, color='red', label='Uncertainty Zone (0.3≤h≤0.7)')
    ax2.axvspan(0.7, 1.0, alpha=0.15, color='green')

    # Reference line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Annotations
    ax2.annotate('GCN WINS\n(structure helps)', xy=(0.15, 2), fontsize=10,
                 ha='center', color='darkgreen', fontweight='bold')
    ax2.annotate('GCN LOSES\n(structure hurts)', xy=(0.5, -15), fontsize=10,
                 ha='center', color='darkred', fontweight='bold')
    ax2.annotate('GCN WINS\n(structure helps)', xy=(0.85, 2), fontsize=10,
                 ha='center', color='darkgreen', fontweight='bold')

    ax2.set_xlabel('Edge Homophily (h)')
    ax2.set_ylabel('GCN Advantage over MLP (%)')
    ax2.set_title('(b) Trust Region of Graph Propagation')
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-22, 8)
    ax2.legend(loc='lower center', fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "spi_correlation.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "spi_correlation.pdf", format='pdf', bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'spi_correlation.png'}")
    plt.close()

def calculate_trust_region_boundaries():
    """
    P0-2: Define Trust Region boundaries quantitatively.
    Calculate exact h range where GNN > MLP and where GNN < MLP.
    """
    data = load_all_data()
    cross_model = [d for d in data if d['source'] == 'cross_model']

    print("\n" + "="*60)
    print("TRUST REGION BOUNDARY ANALYSIS")
    print("="*60)

    # Find transition points for GCN
    print("\nGCN Trust Region:")
    gcn_wins = [(d['h'], d['gcn_advantage']) for d in cross_model if d['gcn_advantage'] > 0]
    gcn_loses = [(d['h'], d['gcn_advantage']) for d in cross_model if d['gcn_advantage'] <= 0]

    print(f"  GCN wins at h: {sorted([h for h, _ in gcn_wins])}")
    print(f"  GCN loses at h: {sorted([h for h, _ in gcn_loses])}")

    # Calculate boundaries
    if gcn_wins and gcn_loses:
        win_h = [h for h, _ in gcn_wins]
        lose_h = [h for h, _ in gcn_loses]

        # Lower boundary (transition from win to lose)
        lower_wins = [h for h in win_h if h < 0.5]
        upper_wins = [h for h in win_h if h > 0.5]

        if lower_wins and lose_h:
            lower_boundary = max(lower_wins)
            print(f"\n  Lower Trust Region: h < {lower_boundary + 0.05:.2f}")

        if upper_wins and lose_h:
            upper_boundary = min(upper_wins)
            print(f"  Upper Trust Region: h > {upper_boundary - 0.05:.2f}")

        mid_zone = [h for h in lose_h]
        print(f"\n  Uncertainty Zone: h ∈ [{min(mid_zone):.1f}, {max(mid_zone):.1f}]")

    # SPI threshold
    print("\n" + "-"*60)
    print("SPI-based Trust Region:")
    print("-"*60)

    # Find SPI threshold where GCN advantage crosses 0
    spi_values = np.array([d['spi'] for d in cross_model])
    gcn_adv = np.array([d['gcn_advantage'] for d in cross_model])

    # Linear interpolation to find zero-crossing
    slope, intercept, _, _, _ = stats.linregress(spi_values, gcn_adv)
    spi_threshold = -intercept / slope if slope != 0 else 0.5

    print(f"  SPI threshold (linear): {spi_threshold:.3f}")
    print(f"  → Use GNN when SPI > {spi_threshold:.2f}")
    print(f"  → Use MLP when SPI < {spi_threshold:.2f}")
    print(f"  → This corresponds to h < {(1 - spi_threshold) / 2:.2f} or h > {(1 + spi_threshold) / 2:.2f}")

    # Return summary
    return {
        'spi_threshold': spi_threshold,
        'uncertainty_zone_h': [min(gcn_loses, key=lambda x: x[0])[0],
                               max(gcn_loses, key=lambda x: x[0])[0]],
        'trust_zone_low': (0, max(lower_wins) if 'lower_wins' in dir() and lower_wins else 0.2),
        'trust_zone_high': (min(upper_wins) if 'upper_wins' in dir() and upper_wins else 0.8, 1.0)
    }

def create_summary_table():
    """Create summary table for paper."""
    data = load_all_data()
    cross_model = [d for d in data if d['source'] == 'cross_model']

    print("\n" + "="*70)
    print("TABLE: SPI and Model Performance Summary")
    print("="*70)
    print(f"{'h':<6} {'SPI':<8} {'GCN(%)':<10} {'GAT(%)':<10} {'SAGE(%)':<10} {'Zone':<15}")
    print("-"*70)

    for d in sorted(cross_model, key=lambda x: x['h']):
        zone = 'Trust' if d['spi'] > 0.4 else 'Uncertainty'
        print(f"{d['h']:<6.1f} {d['spi']:<8.2f} {d['gcn_advantage']:>+8.1f}  "
              f"{d['gat_advantage']:>+8.1f}  {d['sage_advantage']:>+8.1f}  {zone:<15}")

    print("-"*70)

if __name__ == "__main__":
    # Run correlation analysis
    corr_stats = analyze_spi_correlation()

    # Calculate trust region boundaries
    boundaries = calculate_trust_region_boundaries()

    # Create visualizations
    plot_spi_correlation()

    # Create summary table
    create_summary_table()

    # Save results
    results = {
        'correlation': corr_stats,
        'trust_region': boundaries
    }

    output_path = Path(__file__).parent / "spi_correlation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
