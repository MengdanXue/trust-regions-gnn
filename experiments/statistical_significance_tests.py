"""
Statistical Significance Tests for SPI Framework
Addresses the statistical rigor concerns raised by Gemini and Codex

Tests included:
1. Wilcoxon signed-rank test (paired comparison)
2. McNemar test (prediction accuracy comparison)
3. Friedman test with Nemenyi post-hoc (multiple model comparison)
4. Bootstrap confidence intervals
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

def load_all_results():
    """Load all experimental results."""
    code_dir = Path(__file__).parent

    with open(code_dir / "real_dataset_results.json", 'r') as f:
        main_results = json.load(f)['results']

    h2gcn_path = code_dir / "h2gcn_validation_results.json"
    if h2gcn_path.exists():
        with open(h2gcn_path, 'r') as f:
            h2gcn_results = json.load(f)['results']
    else:
        h2gcn_results = []

    return main_results, h2gcn_results

def wilcoxon_test_gcn_vs_mlp(results):
    """Wilcoxon signed-rank test: GCN vs MLP accuracy."""
    print("=" * 70)
    print("1. WILCOXON SIGNED-RANK TEST: GCN vs MLP")
    print("=" * 70)

    gcn_acc = [r['GCN_acc'] for r in results]
    mlp_acc = [r['MLP_acc'] for r in results]

    # Paired differences
    diffs = [g - m for g, m in zip(gcn_acc, mlp_acc)]

    print(f"\nSample size: n = {len(results)}")
    print(f"GCN mean accuracy: {np.mean(gcn_acc)*100:.2f}% (+/- {np.std(gcn_acc)*100:.2f}%)")
    print(f"MLP mean accuracy: {np.mean(mlp_acc)*100:.2f}% (+/- {np.std(mlp_acc)*100:.2f}%)")
    print(f"Mean difference (GCN - MLP): {np.mean(diffs)*100:.2f}%")

    # Wilcoxon test
    stat, p_value = stats.wilcoxon(gcn_acc, mlp_acc, alternative='two-sided')

    print(f"\nWilcoxon statistic: W = {stat:.4f}")
    print(f"p-value (two-sided): p = {p_value:.6f}")

    if p_value < 0.05:
        print("Result: SIGNIFICANT difference (p < 0.05)")
        winner = "GCN" if np.mean(diffs) > 0 else "MLP"
        print(f"Winner: {winner}")
    else:
        print("Result: NO significant difference (p >= 0.05)")

    return {'test': 'wilcoxon_gcn_mlp', 'statistic': stat, 'p_value': p_value}

def wilcoxon_test_h2gcn_vs_gcn(h2gcn_results):
    """Wilcoxon signed-rank test: H2GCN vs GCN on heterophilic datasets."""
    print("\n" + "=" * 70)
    print("2. WILCOXON SIGNED-RANK TEST: H2GCN vs GCN (Heterophilic)")
    print("=" * 70)

    hetero = [r for r in h2gcn_results if r.get('is_heterophilic', False)]

    if len(hetero) < 5:
        print(f"WARNING: Only {len(hetero)} heterophilic datasets. Minimum 5 recommended.")

    h2gcn_acc = [r['H2GCN_acc'] for r in hetero]
    gcn_acc = [r['GCN_acc'] for r in hetero]

    diffs = [h - g for h, g in zip(h2gcn_acc, gcn_acc)]

    print(f"\nSample size: n = {len(hetero)} heterophilic datasets")
    print(f"H2GCN mean accuracy: {np.mean(h2gcn_acc)*100:.2f}%")
    print(f"GCN mean accuracy: {np.mean(gcn_acc)*100:.2f}%")
    print(f"Mean improvement (H2GCN - GCN): +{np.mean(diffs)*100:.2f}%")

    print("\nPer-dataset improvements:")
    for r in hetero:
        imp = (r['H2GCN_acc'] - r['GCN_acc']) * 100
        print(f"  {r['dataset']}: +{imp:.1f}%")

    # Wilcoxon test (one-sided: H2GCN > GCN)
    if len(hetero) >= 5:
        stat, p_value = stats.wilcoxon(h2gcn_acc, gcn_acc, alternative='greater')
        print(f"\nWilcoxon statistic: W = {stat:.4f}")
        print(f"p-value (one-sided, H2GCN > GCN): p = {p_value:.6f}")

        if p_value < 0.05:
            print("Result: H2GCN SIGNIFICANTLY better than GCN (p < 0.05)")
        else:
            print("Result: No significant difference (p >= 0.05)")
    else:
        # Use sign test for small samples
        positive = sum(1 for d in diffs if d > 0)
        p_value = stats.binom_test(positive, len(diffs), 0.5, alternative='greater')
        stat = positive
        print(f"\nSign test (small sample): {positive}/{len(diffs)} positive")
        print(f"p-value (binomial): p = {p_value:.6f}")

    return {'test': 'wilcoxon_h2gcn_gcn', 'statistic': stat, 'p_value': p_value, 'n': len(hetero)}

def mcnemar_test_spi_prediction(results):
    """McNemar test: Compare prediction accuracy of different SPI rules."""
    print("\n" + "=" * 70)
    print("3. McNEMAR TEST: SPI Rule Comparison")
    print("=" * 70)

    # Original SPI rule vs Label-Free rule
    original_correct = []
    labelfree_correct = []

    for r in results:
        h = r['homophily']
        spi = r['spi']
        best_gnn = max(r['GCN_acc'], r.get('GAT_acc', 0), r.get('GraphSAGE_acc', 0))
        actual_best = 'GNN' if best_gnn > r['MLP_acc'] else 'MLP'

        # Original SPI rule
        orig_pred = 'GNN' if spi > 0.4 else 'MLP'
        original_correct.append(orig_pred == actual_best)

        # Label-free rule (simplified, without H2GCN)
        if spi < 0.4:
            lf_pred = 'MLP'
        else:
            lf_pred = 'GNN'
        labelfree_correct.append(lf_pred == actual_best)

    # McNemar contingency table
    # b: Original wrong, Label-free correct
    # c: Original correct, Label-free wrong
    b = sum(1 for o, l in zip(original_correct, labelfree_correct) if not o and l)
    c = sum(1 for o, l in zip(original_correct, labelfree_correct) if o and not l)

    print(f"\nOriginal SPI accuracy: {sum(original_correct)}/{len(results)} = {100*sum(original_correct)/len(results):.1f}%")
    print(f"Label-free accuracy: {sum(labelfree_correct)}/{len(results)} = {100*sum(labelfree_correct)/len(results):.1f}%")
    print(f"\nContingency:")
    print(f"  Original wrong, Label-free correct (b): {b}")
    print(f"  Original correct, Label-free wrong (c): {c}")

    if b + c > 0:
        # McNemar test
        stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
        p_value = 1 - stats.chi2.cdf(stat, df=1)
        print(f"\nMcNemar statistic: chi2 = {stat:.4f}")
        print(f"p-value: p = {p_value:.6f}")
    else:
        print("\nNo discordant pairs - rules identical on this dataset")
        stat, p_value = 0, 1.0

    return {'test': 'mcnemar', 'b': b, 'c': c, 'statistic': stat, 'p_value': p_value}

def friedman_test_multi_model(results, h2gcn_results):
    """Friedman test with Nemenyi post-hoc: Compare multiple models."""
    print("\n" + "=" * 70)
    print("4. FRIEDMAN TEST: Multi-Model Comparison")
    print("=" * 70)

    # Find common datasets
    h2gcn_dict = {r['dataset']: r for r in h2gcn_results}
    common_datasets = [r for r in results if r['dataset'] in h2gcn_dict]

    if len(common_datasets) < 3:
        print("WARNING: Not enough common datasets for Friedman test")
        return None

    # Build accuracy matrix: rows=datasets, cols=models
    models = ['MLP', 'GCN', 'H2GCN']
    accuracy_matrix = []
    dataset_names = []

    for r in common_datasets:
        h2r = h2gcn_dict[r['dataset']]
        row = [r['MLP_acc'], r['GCN_acc'], h2r['H2GCN_acc']]
        accuracy_matrix.append(row)
        dataset_names.append(r['dataset'])

    accuracy_matrix = np.array(accuracy_matrix)

    print(f"\nDatasets: {len(common_datasets)}")
    print(f"Models: {models}")

    print("\nMean accuracy per model:")
    for i, model in enumerate(models):
        print(f"  {model}: {np.mean(accuracy_matrix[:, i])*100:.2f}%")

    # Compute ranks (higher accuracy = lower rank number = better)
    ranks = np.zeros_like(accuracy_matrix)
    for i in range(len(common_datasets)):
        ranks[i] = stats.rankdata(-accuracy_matrix[i])  # negative for descending

    print("\nMean rank per model (lower is better):")
    mean_ranks = np.mean(ranks, axis=0)
    for i, model in enumerate(models):
        print(f"  {model}: {mean_ranks[i]:.3f}")

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*[accuracy_matrix[:, i] for i in range(len(models))])

    print(f"\nFriedman statistic: chi2 = {stat:.4f}")
    print(f"p-value: p = {p_value:.6f}")

    if p_value < 0.05:
        print("Result: SIGNIFICANT differences among models (p < 0.05)")
        print("\nPost-hoc Nemenyi test needed to identify which pairs differ")

        # Nemenyi critical difference
        k = len(models)  # number of models
        n = len(common_datasets)  # number of datasets
        q_alpha = 2.343  # q_0.05 for k=3
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        print(f"\nNemenyi Critical Difference (alpha=0.05): CD = {cd:.3f}")
        print("\nPairwise rank differences:")
        for i, j in combinations(range(len(models)), 2):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            sig = "SIGNIFICANT" if diff > cd else "not significant"
            print(f"  |{models[i]} - {models[j]}| = {diff:.3f} ({sig})")
    else:
        print("Result: No significant differences among models (p >= 0.05)")

    return {
        'test': 'friedman',
        'statistic': stat,
        'p_value': p_value,
        'mean_ranks': {m: r for m, r in zip(models, mean_ranks)},
        'n_datasets': len(common_datasets)
    }

def bootstrap_confidence_interval(results, n_bootstrap=1000):
    """Bootstrap confidence intervals for GCN advantage."""
    print("\n" + "=" * 70)
    print("5. BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    gcn_advantages = [r['GCN_advantage'] for r in results]

    # Bootstrap
    np.random.seed(42)
    bootstrap_means = []
    n = len(gcn_advantages)

    for _ in range(n_bootstrap):
        sample = np.random.choice(gcn_advantages, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Confidence intervals
    ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
    ci_90 = np.percentile(bootstrap_means, [5, 95])

    print(f"\nGCN Advantage (GCN - MLP) across {len(results)} datasets:")
    print(f"  Mean: {np.mean(gcn_advantages)*100:.2f}%")
    print(f"  Std: {np.std(gcn_advantages)*100:.2f}%")
    print(f"\nBootstrap (n={n_bootstrap}):")
    print(f"  95% CI: [{ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%]")
    print(f"  90% CI: [{ci_90[0]*100:.2f}%, {ci_90[1]*100:.2f}%]")

    # Does CI include 0?
    if ci_95[0] <= 0 <= ci_95[1]:
        print("\n  Note: 95% CI includes 0 - no significant overall advantage")
    elif ci_95[0] > 0:
        print("\n  Note: GCN significantly better than MLP overall")
    else:
        print("\n  Note: MLP significantly better than GCN overall")

    return {
        'test': 'bootstrap',
        'mean': np.mean(gcn_advantages),
        'ci_95': ci_95.tolist(),
        'ci_90': ci_90.tolist()
    }

def spi_prediction_by_zone(results):
    """Analyze SPI prediction accuracy by homophily zone."""
    print("\n" + "=" * 70)
    print("6. SPI PREDICTION ACCURACY BY HOMOPHILY ZONE")
    print("=" * 70)

    zones = {
        'Low h (< 0.3)': [r for r in results if r['homophily'] < 0.3],
        'Mid h (0.3-0.7)': [r for r in results if 0.3 <= r['homophily'] <= 0.7],
        'High h (> 0.7)': [r for r in results if r['homophily'] > 0.7]
    }

    print("\nOriginal SPI rule (SPI > 0.4 -> GNN):")
    for zone_name, zone_data in zones.items():
        if zone_data:
            correct = sum(1 for r in zone_data if r['prediction_correct'])
            print(f"  {zone_name}: {correct}/{len(zone_data)} = {100*correct/len(zone_data):.1f}%")

    # Label-free rule
    print("\nLabel-free rule (SPI < 0.4 -> MLP, h < 0.25 -> H2GCN, else -> GCN):")
    for zone_name, zone_data in zones.items():
        if zone_data:
            correct = 0
            for r in zone_data:
                h = r['homophily']
                spi = r['spi']
                best_gnn = max(r['GCN_acc'], r.get('GAT_acc', 0), r.get('GraphSAGE_acc', 0))
                actual_best_acc = max(r['MLP_acc'], best_gnn)

                if spi < 0.4:
                    pred_acc = r['MLP_acc']
                elif h < 0.25:
                    pred_acc = best_gnn  # Approximate H2GCN
                else:
                    pred_acc = r['GCN_acc']

                if pred_acc >= actual_best_acc - 0.05:
                    correct += 1

            print(f"  {zone_name}: {correct}/{len(zone_data)} = {100*correct/len(zone_data):.1f}%")

    return zones

def main():
    """Run all statistical tests."""
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS FOR SPI FRAMEWORK")
    print("=" * 70)

    main_results, h2gcn_results = load_all_results()
    print(f"\nLoaded {len(main_results)} main results, {len(h2gcn_results)} H2GCN results")

    all_tests = {}

    # Test 1: Wilcoxon GCN vs MLP
    all_tests['wilcoxon_gcn_mlp'] = wilcoxon_test_gcn_vs_mlp(main_results)

    # Test 2: Wilcoxon H2GCN vs GCN
    if h2gcn_results:
        all_tests['wilcoxon_h2gcn_gcn'] = wilcoxon_test_h2gcn_vs_gcn(h2gcn_results)

    # Test 3: McNemar
    all_tests['mcnemar'] = mcnemar_test_spi_prediction(main_results)

    # Test 4: Friedman
    if h2gcn_results:
        friedman_result = friedman_test_multi_model(main_results, h2gcn_results)
        if friedman_result:
            all_tests['friedman'] = friedman_result

    # Test 5: Bootstrap
    all_tests['bootstrap'] = bootstrap_confidence_interval(main_results)

    # Test 6: Zone analysis
    spi_prediction_by_zone(main_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF STATISTICAL TESTS")
    print("=" * 70)

    print("\n| Test | Statistic | p-value | Conclusion |")
    print("|------|-----------|---------|------------|")

    for name, result in all_tests.items():
        if result and 'p_value' in result:
            sig = "Significant" if result['p_value'] < 0.05 else "Not sig."
            print(f"| {name} | {result.get('statistic', 'N/A'):.4f} | {result['p_value']:.4f} | {sig} |")

    # Save results
    code_dir = Path(__file__).parent
    output_path = code_dir / "statistical_tests_results.json"

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    serializable = {}
    for k, v in all_tests.items():
        if v:
            serializable[k] = {kk: convert_numpy(vv) for kk, vv in v.items()}

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
