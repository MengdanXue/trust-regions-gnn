# Trust Regions of Graph Propagation

Official code for the paper **"Trust Regions of Graph Propagation: When to Use GNNs and When Not To"**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/MengdanXue/trust-regions-gnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## One-Click Reproduction

```bash
# Quick test (5 minutes)
python scripts/reproduce_paper.py --quick

# Full reproduction (2 hours)
python scripts/reproduce_paper.py --full
```

This generates `results/reproduction/reproduction_report.md` comparing your results with the paper.

## Key Findings

### 1. The U-Shape Discovery

GNN performance follows a **U-shaped pattern** across the homophily spectrum:
- GNNs outperform MLP at **both extremes** (h < 0.3 and h > 0.7)
- GNNs significantly **underperform** in the mid-range (0.3 < h < 0.7) by up to **18%**

This challenges the conventional binary framing of homophilic vs. heterophilic graphs.

### 2. Structural Predictability Index (SPI)

```
SPI = |2h - 1|
```

- Ranges from 0 (maximum uncertainty at h=0.5) to 1 (perfect predictability)
- **R² = 0.82** correlation with GNN advantage (p < 10⁻¹⁷)
- **Quadratic model R² = 0.968** (F-test p = 0.0043, confirms I ∝ SPI² theory)
- Computable in **O(|E|)** time

### 3. Cross-Model Validation

| Model | U-Shape Amplitude | Mid-h Zone Disadvantage |
|-------|-------------------|------------------------|
| GCN | 18.9% | -9.6% |
| GAT | 3.2% | -1.4% |
| GraphSAGE | 0.7% | +0.4% |

**Key Finding**: GraphSAGE is a safe default—wins or ties in 8/9 settings.

### 4. Feature-Pattern Duality

Critical discovery: **the U-shape is a synthetic artifact**. With real features:
- GNN advantage follows a **monotonic** pattern (not U-shaped)
- Crossover occurs at h ≈ 0.5
- Real heterophilic neighbors are feature-orthogonal (noise), not feature-opposite (signal)
- Even H2GCN reduces damage by 22% but cannot reverse monotonicity

### 5. Two-Hop Recovery as Key Diagnostic

The 2-hop label recovery ratio (r = -0.89, p = 0.02) is the strongest predictor:
- **R > 1.5×**: Recoverable heterophily → use heterophily-aware GNNs (H2GCN, LINKX)
- **R < 1×**: Irrecoverable noise → use MLP

### 6. Trust Region Framework

| Region | Condition | Recommendation |
|--------|-----------|----------------|
| Trust Region | h > 0.5 (SPI > 0) | Use GNN |
| Q2 Quadrant | h < 0.5, MLP acc > 65% | Use MLP |
| Uncertain | h < 0.5, MLP acc < 65% | Check 2-hop recovery |

**LOO-CV Accuracy: 100% (19/19 datasets)**

## Installation

```bash
# Clone the repository
git clone https://github.com/MengdanXue/trust-regions-gnn.git
cd trust-regions-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- NumPy >= 1.21
- SciPy >= 1.7
- Matplotlib >= 3.5

## Quick Start

### 1. H-Sweep Experiment (U-Shape Discovery)

```bash
python experiments/h_sweep_experiment.py
```

Generates synthetic graphs with controlled homophily and validates the U-shape pattern.

### 2. Cross-Model Validation

```bash
python experiments/cross_model_hsweep.py
```

Compares U-shape severity across GCN, GAT, and GraphSAGE.

### 3. Semi-Synthetic Experiments (Feature-Pattern Duality)

```bash
python experiments/semi_synthetic_hsweep.py
```

Tests the monotonic pattern with real features + rewired topology.

### 4. Real Dataset Validation (28 datasets)

```bash
python experiments/real_dataset_validation.py
```

Validates on diverse real-world datasets including citation, social, and e-commerce networks.

### 5. Leave-One-Out Cross-Validation

```bash
python experiments/lodo_validation.py
```

Validates generalizability with LOO-CV (100% accuracy on 19 datasets).

## Project Structure

```
trust-regions-gnn/
├── experiments/
│   ├── h_sweep_experiment.py          # Core U-shape experiment
│   ├── cross_model_hsweep.py          # Cross-model validation
│   ├── semi_synthetic_hsweep.py       # Feature-Pattern Duality
│   ├── separability_sweep.py          # Feature separability sweep
│   ├── real_dataset_validation.py     # 28 real-world datasets
│   ├── lodo_validation.py             # Leave-One-Dataset-Out CV
│   ├── two_hop_recovery.py            # 2-hop diagnostic
│   ├── feature_similarity_gap_analysis.py  # r=-0.755 analysis
│   ├── ogb_validation.py              # ogbn-arxiv (170K nodes)
│   └── ogb_products_validation.py     # ogbn-products (2.4M nodes)
├── models/
│   ├── gcn.py                         # GCN implementation
│   ├── gat.py                         # GAT implementation
│   ├── graphsage.py                   # GraphSAGE implementation
│   └── mlp.py                         # MLP baseline
├── utils/
│   ├── data_generation.py             # Synthetic graph generation
│   ├── metrics.py                     # SPI and homophily metrics
│   └── statistical_tests.py           # Wilcoxon, Friedman tests
├── results/                           # Pre-computed experiment results
├── figures/                           # Generated figures
├── configs/                           # Configuration files
├── scripts/
│   └── reproduce_paper.py             # One-click reproduction
├── LICENSE                            # MIT License
├── requirements.txt
└── README.md
```

## Key Functions

### Calculate SPI

```python
def calculate_spi(h):
    """Structural Predictability Index: SPI = |2h - 1|"""
    return abs(2 * h - 1)
```

### Calculate Edge Homophily

```python
def calculate_homophily(edge_index, labels):
    """Calculate edge homophily ratio"""
    src, dst = edge_index
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()
```

### Two-Factor Decision Rule

```python
def should_use_gnn(h, mlp_accuracy, two_hop_ratio=None):
    """
    Two-factor decision framework for model selection.

    Args:
        h: Edge homophily
        mlp_accuracy: MLP baseline accuracy (feature sufficiency)
        two_hop_ratio: h_2 / h (optional, for low-h disambiguation)

    Returns:
        str: 'GNN', 'MLP', or 'Uncertain'
    """
    # Step 1: Trust Region check
    if h > 0.5:
        return 'GNN'  # 100% accuracy in this region

    # Step 2: Q2 Quadrant check (high feature sufficiency)
    if mlp_accuracy > 0.65:
        return 'MLP'  # Features sufficient, structure is noise

    # Step 3: Check 2-hop recovery for low-h, low-FS
    if two_hop_ratio is not None:
        if two_hop_ratio > 1.5:
            return 'GNN'  # Recoverable heterophily
        elif two_hop_ratio < 1.0:
            return 'MLP'  # Irrecoverable noise

    return 'Uncertain'
```

## Reproducing Paper Results

### Table 1: Cross-Model H-Sweep (Zone-wise)

```bash
python experiments/cross_model_hsweep.py
```

Expected output:
| Model | Low h (<0.3) | Mid h (0.3-0.7) | High h (>0.7) |
|-------|--------------|-----------------|---------------|
| GCN | +0.1% | -9.6% | +0.5% |
| GAT | +0.2% | -1.4% | +0.3% |
| GraphSAGE | +0.6% | +0.4% | +0.5% |

### Table 2: LOO-CV Results

```bash
python experiments/lodo_validation.py
```

Expected output:
| Region | Accuracy | Datasets |
|--------|----------|----------|
| Trust Region (h > 0.5) | 100% | 11/11 |
| Q2 (High FS, Low h) | 100% | 4/4 |
| Q4 (Low FS, Low h) | 100% | 4/4 |
| **Overall** | **100%** | **19/19** |

### Statistical Validation

```bash
python experiments/statistical_significance_tests.py
```

Expected output:
- **SPI Correlation**: r = 0.906, 95% CI [0.833, 0.978]
- **R²**: 0.82 (p < 10⁻¹⁷)
- **Quadratic F-test**: p = 0.0043 (significant)
- **Cohen's d at h=0.5**: -9.15 (extremely large effect)
- **Multivariate regression**: homophily p = 0.013 (only significant predictor after controlling for confounders)

## Large-Scale Validation

| Dataset | Nodes | h | SPI | Result |
|---------|-------|---|-----|--------|
| ogbn-arxiv | 170K | 0.66 | 0.31 | GCN -0.6% (Uncertain Zone ✓) |
| ogbn-products | 2.4M | 0.81 | 0.62 | SAGE +17.3% (Trust Region ✓) |
| Penn94 | 39K | 0.51 | 0.02 | GCN +2.1% (boundary ✓) |

## Citation

```bibtex
@article{xue2025trust,
  title={Trust Regions of Graph Propagation: When to Use GNNs and When Not To},
  author={Xue, Mengdan and Yakushin, Alexey V.},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

This work is based on the author's master's thesis at Lomonosov Moscow State University, Faculty of Computational Mathematics and Cybernetics.
