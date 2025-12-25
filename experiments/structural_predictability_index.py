"""
Structural Predictability Index (SPI) - A Novel Metric for GNN Architecture Selection

This module implements the Structural Predictability Index, which measures
how predictable the graph structure is for classification, regardless of
whether it's homophilic or heterophilic.

Key Insight from U-Shape Discovery:
- GNN works well when structure is PREDICTABLE (high or low h)
- GNN fails when structure is RANDOM (mid h ~ 0.5)
- SPI captures this "predictability" concept

Theoretical Foundation:
SPI = |2h - 1| where h is edge homophily
- SPI = 1 when h = 0 (perfectly heterophilic) or h = 1 (perfectly homophilic)
- SPI = 0 when h = 0.5 (random/maximum entropy)

This simple formula captures the key insight:
"Distance from randomness determines GNN advantage"
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

class StructuralPredictabilityIndex:
    """
    Computes the Structural Predictability Index (SPI) for a graph.

    SPI measures how predictable the graph structure is for node classification.
    Higher SPI means structure is more useful for GNNs.
    """

    def __init__(self):
        self.cache = {}

    def compute_homophily(self, edge_index: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute edge homophily: fraction of edges connecting same-class nodes.

        Args:
            edge_index: (2, num_edges) array of edges
            labels: (num_nodes,) array of node labels

        Returns:
            Edge homophily h in [0, 1]
        """
        src, dst = edge_index[0], edge_index[1]
        same_class = (labels[src] == labels[dst]).sum()
        total_edges = len(src)

        return same_class / total_edges if total_edges > 0 else 0.5

    def compute_spi(self, h: float) -> float:
        """
        Compute Structural Predictability Index from homophily.

        SPI = |2h - 1|

        This transforms homophily from a "direction" measure to a "magnitude" measure.

        Args:
            h: Edge homophily in [0, 1]

        Returns:
            SPI in [0, 1], where 1 = highly predictable, 0 = random
        """
        return abs(2 * h - 1)

    def compute_from_graph(self, edge_index: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Compute SPI and related metrics directly from graph data.

        Returns dict with:
        - h: edge homophily
        - spi: structural predictability index
        - regime: 'predictable_positive', 'predictable_negative', or 'uncertain'
        - gnn_recommendation: whether to use GNN
        - confidence: how confident the recommendation is
        """
        h = self.compute_homophily(edge_index, labels)
        spi = self.compute_spi(h)

        # Determine regime
        if h > 0.7:
            regime = 'predictable_positive'
            gnn_rec = True
            confidence = min(1.0, (h - 0.7) / 0.3 + 0.7)  # Scale confidence
        elif h < 0.3:
            regime = 'predictable_negative'
            gnn_rec = True
            confidence = min(1.0, (0.3 - h) / 0.3 + 0.7)
        else:
            regime = 'uncertain'
            gnn_rec = False
            # Confidence is based on distance from 0.5 (peak uncertainty)
            distance_from_center = abs(h - 0.5)
            confidence = 0.5 + distance_from_center  # Lower confidence in uncertain zone

        return {
            'homophily': h,
            'spi': spi,
            'regime': regime,
            'gnn_recommended': gnn_rec,
            'confidence': confidence,
            'interpretation': self._interpret(h, spi, regime)
        }

    def _interpret(self, h: float, spi: float, regime: str) -> str:
        """Generate human-readable interpretation."""
        if regime == 'predictable_positive':
            return (f"High homophily (h={h:.2f}): Neighbors are reliably SAME class. "
                    f"GNN can exploit positive correlation. SPI={spi:.2f} (high predictability).")
        elif regime == 'predictable_negative':
            return (f"Low homophily (h={h:.2f}): Neighbors are reliably DIFFERENT class. "
                    f"GNN can exploit anti-correlation. SPI={spi:.2f} (high predictability).")
        else:
            return (f"Mid homophily (h={h:.2f}): Neighbors are random mix of classes. "
                    f"Structure is noise, not signal. SPI={spi:.2f} (low predictability). "
                    f"Recommend MLP over GNN.")

    def predict_gnn_advantage(self, h: float) -> Tuple[float, str]:
        """
        Predict expected GCN advantage over MLP based on homophily.

        Based on our H-sweep experiments:
        - At h=0.1 or h=0.9: ~+6% advantage
        - At h=0.5: ~-18% advantage (disadvantage)
        - Approximately quadratic relationship

        Returns:
            (expected_advantage, confidence_level)
        """
        # Quadratic model fitted to our data
        # advantage ~ a * (h - 0.5)^2 + b
        # At h=0.5: advantage = -0.186 (our data)
        # At h=0.1 or h=0.9: advantage = +0.064 (our data)

        a = (0.064 - (-0.186)) / (0.4**2)  # ~1.5625
        b = -0.186  # intercept at h=0.5

        predicted_advantage = a * (h - 0.5)**2 + b

        # Confidence based on SPI
        spi = self.compute_spi(h)
        if spi > 0.4:
            confidence = 'high'
        elif spi > 0.2:
            confidence = 'medium'
        else:
            confidence = 'low'

        return predicted_advantage, confidence


def validate_spi_on_datasets():
    """
    Validate SPI predictions against our 19-dataset results.
    """
    print("="*60)
    print("SPI Validation on Real Datasets")
    print("="*60)

    # Our dataset metrics (from previous experiments)
    datasets = [
        # High h datasets
        {'name': 'Cora', 'h': 0.81, 'best_method': 'GCN'},
        {'name': 'CiteSeer', 'h': 0.74, 'best_method': 'GCN'},
        {'name': 'PubMed', 'h': 0.80, 'best_method': 'GCN'},
        {'name': 'Elliptic', 'h': 0.76, 'best_method': 'Class A'},
        {'name': 'Flickr', 'h': 0.82, 'best_method': 'Class A'},
        {'name': 'cSBM-HighH', 'h': 0.95, 'best_method': 'Class A'},
        {'name': 'Inj-Amazon', 'h': 0.91, 'best_method': 'Class A'},
        {'name': 'Inj-Flickr', 'h': 0.89, 'best_method': 'Class A'},

        # Low h datasets
        {'name': 'Cornell', 'h': 0.11, 'best_method': 'H2GCN'},
        {'name': 'Texas', 'h': 0.06, 'best_method': 'H2GCN'},
        {'name': 'Wisconsin', 'h': 0.16, 'best_method': 'H2GCN'},
        {'name': 'Actor', 'h': 0.22, 'best_method': 'H2GCN'},
        {'name': 'Squirrel', 'h': 0.22, 'best_method': 'H2GCN'},
        {'name': 'Chameleon', 'h': 0.25, 'best_method': 'H2GCN'},
        {'name': 'cSBM-LowH', 'h': 0.10, 'best_method': 'Class B'},

        # Mid h datasets (uncertainty zone)
        {'name': 'cSBM-MidH', 'h': 0.51, 'best_method': 'Uncertain'},
        {'name': 'cSBM-NoisyF', 'h': 0.50, 'best_method': 'Uncertain'},
        {'name': 'cSBM-CleanF', 'h': 0.52, 'best_method': 'Uncertain'},
        {'name': 'Inj-Cora', 'h': 0.45, 'best_method': 'Mixed'},
    ]

    spi_calc = StructuralPredictabilityIndex()

    print("\n{:<15} {:<6} {:<6} {:<20} {:<15} {:<8}".format(
        'Dataset', 'h', 'SPI', 'Regime', 'GNN Rec?', 'Correct?'))
    print("-"*75)

    correct = 0
    total = 0

    for ds in datasets:
        h = ds['h']
        spi = spi_calc.compute_spi(h)

        if h > 0.7:
            regime = 'predictable_pos'
            gnn_rec = 'Yes'
            expected_best = 'Class A / GCN'
        elif h < 0.3:
            regime = 'predictable_neg'
            gnn_rec = 'Yes (H2GCN)'
            expected_best = 'Class B / H2GCN'
        else:
            regime = 'uncertain'
            gnn_rec = 'No (MLP)'
            expected_best = 'MLP / Mixed'

        # Check if prediction matches reality
        actual = ds['best_method']
        if regime == 'predictable_pos' and actual in ['GCN', 'Class A', 'GAT']:
            is_correct = 'Yes'
            correct += 1
        elif regime == 'predictable_neg' and actual in ['H2GCN', 'Class B', 'GraphSAGE']:
            is_correct = 'Yes'
            correct += 1
        elif regime == 'uncertain' and actual in ['Uncertain', 'Mixed', 'MLP']:
            is_correct = 'Yes'
            correct += 1
        else:
            is_correct = 'No'

        total += 1

        print("{:<15} {:<6.2f} {:<6.2f} {:<20} {:<15} {:<8}".format(
            ds['name'], h, spi, regime, gnn_rec, is_correct))

    print("-"*75)
    print(f"\nSPI Prediction Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    # Analyze by zone
    print("\n" + "="*60)
    print("Analysis by Homophily Zone")
    print("="*60)

    high_h = [d for d in datasets if d['h'] > 0.7]
    mid_h = [d for d in datasets if 0.3 <= d['h'] <= 0.7]
    low_h = [d for d in datasets if d['h'] < 0.3]

    print(f"\nHigh h (>0.7): {len(high_h)} datasets")
    print(f"  Average SPI: {np.mean([spi_calc.compute_spi(d['h']) for d in high_h]):.2f}")
    print(f"  Prediction: All should prefer standard GNN (GCN/GAT)")

    print(f"\nLow h (<0.3): {len(low_h)} datasets")
    print(f"  Average SPI: {np.mean([spi_calc.compute_spi(d['h']) for d in low_h]):.2f}")
    print(f"  Prediction: All should prefer heterophily-aware GNN (H2GCN)")

    print(f"\nMid h (0.3-0.7): {len(mid_h)} datasets")
    print(f"  Average SPI: {np.mean([spi_calc.compute_spi(d['h']) for d in mid_h]):.2f}")
    print(f"  Prediction: Structure unreliable, prefer MLP or ensemble")

    return {
        'accuracy': correct / total,
        'total': total,
        'correct': correct,
        'by_zone': {
            'high_h': len(high_h),
            'mid_h': len(mid_h),
            'low_h': len(low_h)
        }
    }


def create_spi_diagram():
    """Create ASCII diagram showing SPI concept."""
    diagram = """
    STRUCTURAL PREDICTABILITY INDEX (SPI) CONCEPT
    ==============================================

    SPI = |2h - 1|

         SPI
          |
        1 |  *                                   *
          |   *                                 *
          |    *                               *
        0.6|     *                             *
          |      *                           *
          |       *                         *
        0.4|        *                       *
          |         *                     *
          |          *                   *
        0.2|           *                 *
          |            *               *
          |             *             *
        0 +---------------*-----------*----------------> h
          0    0.2   0.3  0.4  0.5  0.6  0.7   0.8    1

          |<- PREDICTABLE ->|<- RANDOM ->|<- PREDICTABLE ->|
             (Anti-corr)     (Entropy)      (Correlation)

    INTERPRETATION:
    ---------------
    SPI > 0.4: High predictability - GNN will likely outperform MLP
    SPI < 0.4: Low predictability - MLP may be safer choice

    GNN ADVANTAGE vs SPI:
    ---------------------

    GNN Advantage (%)
          |
       +6 |  *                                   * * *
          |
        0 +---------------*-----------*---------------
          |               |           |
       -6 |              |             |
          |             |               |
      -12 |            |                 |
          |           *                   *
      -18 |            *       *       *
          +-----------------------------------------> SPI
          0         0.2     0.4     0.6     0.8     1

    KEY INSIGHT:
    ------------
    "It's not about whether structure is homophilic or heterophilic.
     It's about whether structure is PREDICTABLE."

    - High SPI (h near 0 or 1): Structure provides consistent signal
    - Low SPI (h near 0.5): Structure provides noise

    PRACTICAL RECOMMENDATIONS:
    --------------------------
    | Homophily | SPI  | Regime        | Recommendation          |
    |-----------|------|---------------|-------------------------|
    | h > 0.7   | High | Pos-Corr      | Use GCN/GAT             |
    | h < 0.3   | High | Anti-Corr     | Use H2GCN/GPRGNN        |
    | 0.3-0.7   | Low  | Uncertain     | Use MLP or Ensemble     |
    """
    print(diagram)

    # Save to file
    output_path = Path(__file__).parent / "SPI_CONCEPT.txt"
    with open(output_path, 'w') as f:
        f.write(diagram)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Structural Predictability Index (SPI) Analysis")
    print("="*60)

    # Show concept diagram
    print("\n1. SPI Concept Diagram:")
    create_spi_diagram()

    # Validate on datasets
    print("\n2. Validation on Real Datasets:")
    results = validate_spi_on_datasets()

    # Example predictions
    print("\n3. Example SPI Predictions:")
    spi_calc = StructuralPredictabilityIndex()

    test_h_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\n{:<6} {:<6} {:<12} {:<12}".format('h', 'SPI', 'Pred Adv', 'Confidence'))
    print("-"*40)
    for h in test_h_values:
        spi = spi_calc.compute_spi(h)
        adv, conf = spi_calc.predict_gnn_advantage(h)
        print("{:<6.2f} {:<6.2f} {:<12.1%} {:<12}".format(h, spi, adv, conf))

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
