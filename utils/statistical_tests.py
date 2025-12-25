"""
Statistical Tests for Trust Regions Validation
Following Demsar (2006) guidelines for classifier comparison
"""

import numpy as np
from scipy import stats


def pearson_correlation(x, y):
    """
    Pearson correlation coefficient.

    Args:
        x, y: Arrays of values

    Returns:
        (r, p_value)
    """
    r, p = stats.pearsonr(x, y)
    return r, p


def spearman_correlation(x, y):
    """
    Spearman rank correlation (more robust).

    Args:
        x, y: Arrays of values

    Returns:
        (rho, p_value)
    """
    rho, p = stats.spearmanr(x, y)
    return rho, p


def wilcoxon_signed_rank(x, y, alternative='two-sided'):
    """
    Wilcoxon signed-rank test for paired samples.

    Use for comparing two models on same datasets.

    Args:
        x: Scores for model 1
        y: Scores for model 2
        alternative: 'two-sided', 'greater', 'less'

    Returns:
        (statistic, p_value)
    """
    stat, p = stats.wilcoxon(x, y, alternative=alternative)
    return stat, p


def friedman_test(*groups):
    """
    Friedman test for multiple related samples.

    Use for comparing 3+ models across multiple datasets.

    Args:
        *groups: Multiple arrays of scores

    Returns:
        (chi_squared, p_value)
    """
    stat, p = stats.friedmanchisquare(*groups)
    return stat, p


def cohens_d(x, y):
    """
    Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        x, y: Arrays of values

    Returns:
        Cohen's d
    """
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std


def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        adjusted_alpha, which_significant (boolean array)
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    significant = np.array(p_values) < adjusted_alpha
    return adjusted_alpha, significant


def confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval for mean.

    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)

    Returns:
        (mean, lower, upper)
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def run_full_statistical_analysis(spi_values, gnn_advantages,
                                   trust_region_results, uncertain_results,
                                   model_results_dict):
    """
    Run comprehensive statistical analysis for paper.

    Args:
        spi_values: Array of SPI values
        gnn_advantages: Array of GNN-MLP differences
        trust_region_results: Dict with 'gnn' and 'mlp' arrays for trust region
        uncertain_results: Dict with 'gnn' and 'mlp' arrays for uncertain zone
        model_results_dict: Dict of model_name -> accuracy array

    Returns:
        Dict with all test results
    """
    results = {}

    # 1. SPI Correlation
    r_pearson, p_pearson = pearson_correlation(spi_values, gnn_advantages)
    rho_spearman, p_spearman = spearman_correlation(spi_values, gnn_advantages)
    results['spi_correlation'] = {
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_rho': rho_spearman,
        'spearman_p': p_spearman,
        'r_squared': r_pearson ** 2
    }

    # 2. Trust Region Comparison
    w_trust, p_trust = wilcoxon_signed_rank(
        trust_region_results['gnn'],
        trust_region_results['mlp'],
        alternative='greater'
    )
    d_trust = cohens_d(trust_region_results['gnn'], trust_region_results['mlp'])
    results['trust_region'] = {
        'wilcoxon_w': w_trust,
        'wilcoxon_p': p_trust,
        'cohens_d': d_trust
    }

    # 3. Uncertain Zone Comparison
    w_uncertain, p_uncertain = wilcoxon_signed_rank(
        uncertain_results['gnn'],
        uncertain_results['mlp'],
        alternative='less'
    )
    d_uncertain = cohens_d(uncertain_results['gnn'], uncertain_results['mlp'])
    results['uncertain_zone'] = {
        'wilcoxon_w': w_uncertain,
        'wilcoxon_p': p_uncertain,
        'cohens_d': d_uncertain
    }

    # 4. Multi-model Comparison (Friedman)
    model_arrays = list(model_results_dict.values())
    if len(model_arrays) >= 3:
        chi2, p_friedman = friedman_test(*model_arrays)
        results['multi_model'] = {
            'friedman_chi2': chi2,
            'friedman_p': p_friedman
        }

    # 5. Bonferroni Correction
    all_p_values = [p_pearson, p_trust, p_uncertain]
    if 'multi_model' in results:
        all_p_values.append(results['multi_model']['friedman_p'])

    adj_alpha, significant = bonferroni_correction(all_p_values)
    results['bonferroni'] = {
        'adjusted_alpha': adj_alpha,
        'all_significant': all(significant)
    }

    return results


def print_statistical_summary(results):
    """Print formatted statistical summary."""
    print("=" * 60)
    print("Statistical Significance Summary")
    print("=" * 60)

    print("\n--- SPI Correlation (N=45) ---")
    spi = results['spi_correlation']
    print(f"Pearson r = {spi['pearson_r']:.4f} (p = {spi['pearson_p']:.2e})")
    print(f"Spearman rho = {spi['spearman_rho']:.4f} (p = {spi['spearman_p']:.2e})")
    print(f"R-squared = {spi['r_squared']:.4f}")

    print("\n--- Trust Region: GCN vs MLP ---")
    tr = results['trust_region']
    print(f"Wilcoxon W = {tr['wilcoxon_w']:.0f} (p = {tr['wilcoxon_p']:.4f})")
    print(f"Cohen's d = {tr['cohens_d']:.2f}")

    print("\n--- Uncertain Zone: GCN vs MLP ---")
    uz = results['uncertain_zone']
    print(f"Wilcoxon W = {uz['wilcoxon_w']:.0f} (p = {uz['wilcoxon_p']:.4f})")
    print(f"Cohen's d = {uz['cohens_d']:.2f}")

    if 'multi_model' in results:
        print("\n--- Multi-Model Comparison ---")
        mm = results['multi_model']
        print(f"Friedman chi-squared = {mm['friedman_chi2']:.2f} (p = {mm['friedman_p']:.4f})")

    print("\n--- Bonferroni Correction ---")
    bf = results['bonferroni']
    print(f"Adjusted alpha = {bf['adjusted_alpha']:.4f}")
    print(f"All tests significant: {bf['all_significant']}")
    print("=" * 60)
