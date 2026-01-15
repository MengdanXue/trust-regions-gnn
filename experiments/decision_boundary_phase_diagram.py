"""
Decision Boundary Phase Diagram for Trust Regions Paper
Creates h vs 2-hop recovery scatter plot with decision boundaries
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Dataset data: (name, h, 2-hop recovery ratio, winner)
# winner: 'GNN' = standard GNN wins, 'H2GCN' = heterophily-aware wins, 'MLP' = MLP wins
datasets = [
    # High homophily datasets (h > 0.5) - GNN wins
    ('Cora', 0.81, 0.90, 'GNN'),
    ('CiteSeer', 0.74, 1.01, 'GNN'),
    ('PubMed', 0.80, 0.95, 'GNN'),
    ('Amazon-Comp.', 0.78, 0.92, 'GNN'),
    ('Amazon-Photo', 0.83, 0.94, 'GNN'),
    ('Coauthor-CS', 0.81, 0.93, 'GNN'),
    ('Coauthor-Phys.', 0.93, 0.96, 'GNN'),
    ('DBLP', 0.83, 0.91, 'GNN'),
    ('Questions', 0.84, 0.95, 'GNN'),

    # Mid homophily - Mixed
    ('ogbn-arxiv', 0.655, 1.05, 'MLP'),
    ('Tolokers', 0.595, 1.02, 'Tie'),
    ('Minesweeper', 0.683, 1.08, 'Tie'),
    ('Amazon-ratings', 0.38, 1.12, 'GNN'),

    # Low homophily - WebKB (recoverable, R > 1.5)
    ('Texas', 0.108, 5.26, 'H2GCN'),
    ('Wisconsin', 0.196, 2.15, 'H2GCN'),
    ('Cornell', 0.131, 2.99, 'H2GCN'),

    # Low homophily - Wikipedia (irrecoverable, R < 1)
    ('Actor', 0.219, 0.96, 'MLP'),
    ('Chameleon', 0.235, 0.97, 'MLP'),
    ('Squirrel', 0.224, 0.88, 'MLP'),
    ('Roman-empire', 0.047, 0.85, 'MLP'),
]

# Extract data
names = [d[0] for d in datasets]
h_values = np.array([d[1] for d in datasets])
recovery_ratios = np.array([d[2] for d in datasets])
winners = [d[3] for d in datasets]

# Color mapping
color_map = {'GNN': '#2ecc71', 'H2GCN': '#f39c12', 'MLP': '#e74c3c', 'Tie': '#3498db'}
colors = [color_map[w] for w in winners]

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Draw decision boundary regions (background)
# Region 1: h > 0.5 (Trust GNN) - light green
ax.add_patch(Rectangle((0.5, 0), 0.5, 6, facecolor='#d5f4e6', alpha=0.5, zorder=0))
# Region 2: h < 0.5, R > 1.5 (Use H2GCN) - light orange
ax.add_patch(Rectangle((0, 1.5), 0.5, 4.5, facecolor='#fdebd0', alpha=0.5, zorder=0))
# Region 3: h < 0.5, R < 1.5 (Use MLP) - light red
ax.add_patch(Rectangle((0, 0), 0.5, 1.5, facecolor='#fadbd8', alpha=0.5, zorder=0))

# Plot scatter points
for i, (name, h, r, w) in enumerate(datasets):
    marker = 'o' if w == 'GNN' else ('s' if w == 'H2GCN' else ('^' if w == 'MLP' else 'D'))
    size = 150 if w != 'Tie' else 100
    ax.scatter(h, r, c=color_map[w], s=size, marker=marker, edgecolors='black',
               linewidths=1, zorder=3, alpha=0.9)

# Add dataset labels for key points
label_offsets = {
    'Texas': (0.02, 0.3),
    'Wisconsin': (0.02, 0.15),
    'Cornell': (0.02, 0.2),
    'Actor': (0.02, -0.08),
    'Chameleon': (0.02, -0.08),
    'Squirrel': (-0.06, -0.08),
    'Roman-empire': (0.02, -0.08),
    'Cora': (0.02, -0.06),
    'ogbn-arxiv': (0.02, 0.05),
}

for name, h, r, w in datasets:
    if name in label_offsets:
        offset = label_offsets[name]
        ax.annotate(name, (h, r), xytext=(h + offset[0], r + offset[1]),
                   fontsize=8, alpha=0.8)

# Draw decision boundaries
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label='h = 0.5 boundary')
ax.axhline(y=1.5, color='gray', linestyle=':', linewidth=2, xmin=0, xmax=0.5, label='R = 1.5x threshold')

# Add region labels
ax.text(0.75, 5.5, 'Trust Region\n(Use GNN)', fontsize=11, ha='center',
        fontweight='bold', color='#27ae60')
ax.text(0.25, 4.5, 'Recoverable\n(Use H2GCN)', fontsize=11, ha='center',
        fontweight='bold', color='#e67e22')
ax.text(0.25, 0.7, 'Irrecoverable\n(Use MLP)', fontsize=11, ha='center',
        fontweight='bold', color='#c0392b')

# Axis settings
ax.set_xlabel('Homophily (h)', fontsize=12)
ax.set_ylabel('2-Hop Recovery Ratio (R = h₂/h₁)', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 6)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='GNN wins'),
    mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='H2GCN wins'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='MLP wins'),
    mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Tie'),
    plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='h = 0.5'),
    plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='R = 1.5x'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Title
ax.set_title('Decision Boundary Phase Diagram: Two-Factor Architecture Selection',
             fontsize=13, fontweight='bold')

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('figures/decision_boundary_phase_diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/decision_boundary_phase_diagram.png', dpi=300, bbox_inches='tight')
print("Decision Boundary Phase Diagram saved to figures/")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total datasets: {len(datasets)}")

# Count by region
high_h = [(n, h, r, w) for n, h, r, w in datasets if h > 0.5]
low_h_recoverable = [(n, h, r, w) for n, h, r, w in datasets if h <= 0.5 and r > 1.5]
low_h_irrecoverable = [(n, h, r, w) for n, h, r, w in datasets if h <= 0.5 and r <= 1.5]

print(f"\nHigh-h region (h > 0.5): {len(high_h)} datasets")
gnn_wins_high = sum(1 for _, _, _, w in high_h if w == 'GNN' or w == 'Tie')
print(f"  GNN/Tie wins: {gnn_wins_high}/{len(high_h)} = {100*gnn_wins_high/len(high_h):.0f}%")

print(f"\nLow-h Recoverable (h ≤ 0.5, R > 1.5): {len(low_h_recoverable)} datasets")
h2gcn_wins = sum(1 for _, _, _, w in low_h_recoverable if w == 'H2GCN')
print(f"  H2GCN wins: {h2gcn_wins}/{len(low_h_recoverable)} = {100*h2gcn_wins/len(low_h_recoverable):.0f}%")

print(f"\nLow-h Irrecoverable (h ≤ 0.5, R ≤ 1.5): {len(low_h_irrecoverable)} datasets")
mlp_wins = sum(1 for _, _, _, w in low_h_irrecoverable if w == 'MLP')
print(f"  MLP wins: {mlp_wins}/{len(low_h_irrecoverable)} = {100*mlp_wins/len(low_h_irrecoverable):.0f}%")

# Overall accuracy
correct = gnn_wins_high + h2gcn_wins + mlp_wins
total = len(datasets)
print(f"\n=== Overall Two-Factor Accuracy: {correct}/{total} = {100*correct/total:.1f}% ===")
