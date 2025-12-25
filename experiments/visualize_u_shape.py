"""
U-Shape Visualization Script
Creates Figure 1: The U-Shaped Law of GNN Performance

This script generates publication-quality visualizations of the H-Sweep experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

def load_results():
    """Load H-sweep experiment results."""
    results_path = Path(__file__).parent / "h_sweep_v2_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_u_shape_main():
    """Create the main U-shape figure."""
    data = load_results()
    results = data['results']

    h_values = [r['h_actual'] for r in results]
    gcn_acc = [r['gcn_acc'] * 100 for r in results]
    mlp_acc = [r['mlp_acc'] * 100 for r in results]
    gcn_advantage = [r['gcn_advantage'] * 100 for r in results]
    gcn_std = [r['gcn_std'] * 100 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Accuracy comparison
    ax1 = axes[0]
    ax1.plot(h_values, gcn_acc, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='GCN')
    ax1.fill_between(h_values,
                     [g - s for g, s in zip(gcn_acc, gcn_std)],
                     [g + s for g, s in zip(gcn_acc, gcn_std)],
                     alpha=0.2, color='#2E86AB')
    ax1.axhline(y=mlp_acc[0], color='#E94F37', linestyle='--', linewidth=2, label='MLP (baseline)')

    # Shade the zones
    ax1.axvspan(0, 0.3, alpha=0.1, color='green', label='Predictable (low h)')
    ax1.axvspan(0.3, 0.7, alpha=0.1, color='red', label='Uncertain (mid h)')
    ax1.axvspan(0.7, 1.0, alpha=0.1, color='green')

    ax1.set_xlabel('Edge Homophily (h)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) GCN vs MLP Accuracy Across Homophily Spectrum')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(70, 102)
    ax1.legend(loc='lower right')
    ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Right plot: GCN Advantage (U-shape)
    ax2 = axes[1]
    colors = ['#2E86AB' if adv > 0 else '#E94F37' for adv in gcn_advantage]
    ax2.bar(h_values, gcn_advantage, width=0.08, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Annotate key points
    min_idx = np.argmin(gcn_advantage)
    ax2.annotate(f'Valley: {gcn_advantage[min_idx]:.1f}%',
                 xy=(h_values[min_idx], gcn_advantage[min_idx]),
                 xytext=(h_values[min_idx] + 0.1, gcn_advantage[min_idx] - 3),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    # Shade zones
    ax2.axvspan(0, 0.3, alpha=0.1, color='green')
    ax2.axvspan(0.3, 0.7, alpha=0.1, color='red')
    ax2.axvspan(0.7, 1.0, alpha=0.1, color='green')

    # Add zone labels
    ax2.text(0.15, 8, 'GCN\nWins', ha='center', fontsize=10, fontweight='bold', color='#2E86AB')
    ax2.text(0.5, 8, 'MLP\nWins', ha='center', fontsize=10, fontweight='bold', color='#E94F37')
    ax2.text(0.85, 8, 'GCN\nWins', ha='center', fontsize=10, fontweight='bold', color='#2E86AB')

    ax2.set_xlabel('Edge Homophily (h)')
    ax2.set_ylabel('GCN Advantage over MLP (%)')
    ax2.set_title('(b) The U-Shaped Law: GCN Advantage vs Homophily')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-25, 15)
    ax2.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "figures" / "u_shape_main.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PDF for paper
    pdf_path = Path(__file__).parent / "figures" / "u_shape_main.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.show()

def plot_u_shape_conceptual():
    """Create a conceptual diagram explaining the U-shape."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Smooth U-shape curve
    h = np.linspace(0, 1, 100)
    # Quadratic-like function centered at 0.5
    advantage = 6 * (4 * (h - 0.5)**2 - 0.4)

    ax.plot(h, advantage, 'b-', linewidth=3, label='GCN Advantage')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.fill_between(h, advantage, 0, where=(advantage > 0), alpha=0.3, color='green', label='GCN preferred')
    ax.fill_between(h, advantage, 0, where=(advantage < 0), alpha=0.3, color='red', label='MLP preferred')

    # Zone annotations
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.7)

    # Text boxes for each zone
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')

    ax.text(0.15, 7, 'Heterophilic Zone\n(h < 0.3)\n\nNeighbors reliably\nDIFFERENT class\n\nGCN exploits\nanti-correlation',
            ha='center', va='center', fontsize=10, bbox=props)

    ax.text(0.5, -12, 'Uncertainty Zone\n(0.3 < h < 0.7)\n\nNeighbors are\nRANDOM mix\n\nAggregation = noise\nMLP is safer',
            ha='center', va='center', fontsize=10, bbox=props)

    ax.text(0.85, 7, 'Homophilic Zone\n(h > 0.7)\n\nNeighbors reliably\nSAME class\n\nGCN exploits\ncorrelation',
            ha='center', va='center', fontsize=10, bbox=props)

    # Title and labels
    ax.set_xlabel('Edge Homophily (h)', fontsize=14)
    ax.set_ylabel('GCN Advantage over MLP (%)', fontsize=14)
    ax.set_title('The Structural Predictability Principle:\n"It\'s Not About Homophily Being Good or Bad - It\'s About Being PREDICTABLE"',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-20, 12)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "figures" / "u_shape_conceptual.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    pdf_path = Path(__file__).parent / "figures" / "u_shape_conceptual.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.show()

def plot_zone_summary():
    """Create a summary bar chart by zone."""
    data = load_results()

    # Calculate zone averages
    zones = {
        'Low h\n(h < 0.3)': {'gcn': [], 'mlp': []},
        'Mid h\n(0.3-0.7)': {'gcn': [], 'mlp': []},
        'High h\n(h > 0.7)': {'gcn': [], 'mlp': []}
    }

    for r in data['results']:
        h = r['h_actual']
        if h < 0.3:
            zones['Low h\n(h < 0.3)']['gcn'].append(r['gcn_acc'])
            zones['Low h\n(h < 0.3)']['mlp'].append(r['mlp_acc'])
        elif h < 0.7:
            zones['Mid h\n(0.3-0.7)']['gcn'].append(r['gcn_acc'])
            zones['Mid h\n(0.3-0.7)']['mlp'].append(r['mlp_acc'])
        else:
            zones['High h\n(h > 0.7)']['gcn'].append(r['gcn_acc'])
            zones['High h\n(h > 0.7)']['mlp'].append(r['mlp_acc'])

    # Compute means
    zone_names = list(zones.keys())
    gcn_means = [np.mean(zones[z]['gcn']) * 100 for z in zone_names]
    mlp_means = [np.mean(zones[z]['mlp']) * 100 for z in zone_names]

    x = np.arange(len(zone_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, gcn_means, width, label='GCN', color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x + width/2, mlp_means, width, label='MLP', color='#E94F37', edgecolor='black')

    # Add value labels
    for bar, val in zip(bars1, gcn_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, mlp_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add winner annotations
    winners = ['GCN\n(+5.9%)', 'MLP\n(+10.0%)', 'GCN\n(+5.3%)']
    for i, (name, winner) in enumerate(zip(zone_names, winners)):
        color = '#2E86AB' if 'GCN' in winner else '#E94F37'
        ax.text(i, 68, f'Winner:\n{winner}', ha='center', fontsize=10, fontweight='bold', color=color)

    ax.set_ylabel('Average Accuracy (%)', fontsize=14)
    ax.set_title('Zone Summary: GCN vs MLP Performance by Homophily Region', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, fontsize=12)
    ax.set_ylim(60, 105)
    ax.legend(loc='upper right', fontsize=12)
    ax.axhline(y=93.4, color='gray', linestyle=':', alpha=0.7, label='MLP baseline')

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "figures" / "zone_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()

def create_ascii_u_shape():
    """Create ASCII art version of U-shape for terminal/markdown."""
    ascii_art = """
    GCN Advantage over MLP (%)
    |
 +8%|  *                                              * * * *
    |   *                                           *
 +4%|    *                                        *
    |                                            *
  0%|------*----------------------------------*-----------
    |       \\                                /
 -4%|        \\                              /
    |         \\                            /
 -8%|          \\                          /
    |           \\                        /
-12%|            \\                      /
    |             \\                    /
-16%|              \\        *        /
    |               \\      / \\      /
-20%|                *    /   *    /
    |                 \\  /     \\  /
    +-------------------------------------------------> h
    0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0

    |<-- GCN WINS -->|<-- MLP WINS -->|<-- GCN WINS -->|
       (Predictable      (Uncertain)      (Predictable
        Anti-corr)                          Corr)

    KEY INSIGHT: It's not about homophily being "good" or "bad"
                 It's about structure being PREDICTABLE.
    """
    print(ascii_art)

    # Save to file
    output_path = Path(__file__).parent / "figures" / "u_shape_ascii.txt"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(ascii_art)
    print(f"\nSaved ASCII art to: {output_path}")

if __name__ == "__main__":
    print("="*60)
    print("U-Shape Visualization Generator")
    print("="*60)

    # Create ASCII version first (always works)
    print("\n1. Creating ASCII visualization...")
    create_ascii_u_shape()

    # Try matplotlib visualizations
    try:
        print("\n2. Creating main U-shape figure...")
        plot_u_shape_main()

        print("\n3. Creating conceptual diagram...")
        plot_u_shape_conceptual()

        print("\n4. Creating zone summary...")
        plot_zone_summary()

        print("\n" + "="*60)
        print("All visualizations created successfully!")
        print("="*60)
    except Exception as e:
        print(f"\nMatplotlib visualization failed: {e}")
        print("ASCII visualization was created successfully.")
