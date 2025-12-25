"""
Utility functions for Trust Regions experiments
"""

from .metrics import (
    calculate_spi,
    calculate_edge_homophily,
    calculate_node_homophily,
    should_use_gnn,
    get_trust_region,
    calculate_gnn_advantage
)

from .data_generation import (
    generate_csbm_graph,
    generate_h_sweep_datasets
)

from .statistical_tests import (
    pearson_correlation,
    spearman_correlation,
    wilcoxon_signed_rank,
    friedman_test,
    cohens_d,
    bonferroni_correction,
    confidence_interval
)

__all__ = [
    # Metrics
    'calculate_spi',
    'calculate_edge_homophily',
    'calculate_node_homophily',
    'should_use_gnn',
    'get_trust_region',
    'calculate_gnn_advantage',
    # Data generation
    'generate_csbm_graph',
    'generate_h_sweep_datasets',
    # Statistical tests
    'pearson_correlation',
    'spearman_correlation',
    'wilcoxon_signed_rank',
    'friedman_test',
    'cohens_d',
    'bonferroni_correction',
    'confidence_interval',
]
