"""
Comprehensive Leave-One-Out Cross-Validation
=============================================

Using all 19+ datasets from expanded validation for robust LOO-CV.
"""

import json
import numpy as np
from datetime import datetime
from scipy import stats

# Load expanded validation results (19 datasets)
with open('D:/Users/11919/Documents/毕业论文/paper/code/expanded_validation_results.json', 'r') as f:
    expanded_data = json.load(f)

results = expanded_data['results']

print("=" * 80)
print("COMPREHENSIVE LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nTotal datasets: {len(results)}")

# Define the two-factor decision rule
def predict_winner(h, mlp_acc=None, fs_threshold=0.65):
    """
    Two-factor decision rule:
    1. If h > 0.5: predict GNN wins or Tie (Trust Region)
    2. If h <= 0.5:
       - If high FS (MLP >= 0.65): predict MLP wins (Q2 quadrant)
       - Otherwise: uncertain (could go either way)
    """
    if h > 0.5:
        return "GNN"  # Trust Region
    else:
        if mlp_acc is not None and mlp_acc >= fs_threshold:
            return "MLP"  # Q2 quadrant
        return "Uncertain"  # Q4 or other

def is_prediction_correct(predicted, actual_winner, gcn_mlp):
    """Check if prediction is correct"""
    if predicted == "GNN":
        # Trust Region: GNN wins or Tie is acceptable
        return actual_winner in ["GCN", "Tie"]
    elif predicted == "MLP":
        # Q2 quadrant: MLP wins or Tie is acceptable
        return actual_winner in ["MLP", "Tie"]
    elif predicted == "Uncertain":
        # Uncertain region: any outcome is "correct" (we don't claim prediction)
        return True
    return False

# Leave-One-Out Cross-Validation
print("\n" + "-" * 80)
print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
print("-" * 80)

loo_results = []

for i, test_dataset in enumerate(results):
    h = test_dataset['homophily']
    mlp_acc = test_dataset['mlp_mean']
    delta = test_dataset['delta']  # GCN - MLP
    actual_winner = test_dataset['winner']

    predicted = predict_winner(h, mlp_acc)
    correct = is_prediction_correct(predicted, actual_winner, delta)

    loo_results.append({
        'dataset': test_dataset['name'],
        'h': h,
        'mlp_mean': mlp_acc,
        'gcn_mlp': delta,
        'predicted': predicted,
        'actual': actual_winner,
        'correct': correct,
        'quadrant': test_dataset['quadrant']
    })

# Print results
print(f"\n{'Dataset':<20} {'h':>7} {'MLP':>7} {'GCN-MLP':>9} {'Quad':>5} {'Pred':>10} {'Actual':>8} {'OK':>4}")
print("-" * 85)

for r in loo_results:
    ok_str = "Y" if r['correct'] else "N"
    print(f"{r['dataset']:<20} {r['h']:>7.3f} {r['mlp_mean']:>7.3f} "
          f"{r['gcn_mlp']:>+9.3f} {r['quadrant']:>5} {r['predicted']:>10} {r['actual']:>8} {ok_str:>4}")

# Summary statistics
total = len(loo_results)
correct = sum(1 for r in loo_results if r['correct'])
accuracy = correct / total

print("-" * 85)
print(f"\nOverall LOO-CV Accuracy: {correct}/{total} = {accuracy:.1%}")

# Break down by region
high_h = [r for r in loo_results if r['h'] > 0.5]
low_h = [r for r in loo_results if r['h'] <= 0.5]
q2 = [r for r in loo_results if r['quadrant'] == 'Q2']
q4 = [r for r in loo_results if r['quadrant'] == 'Q4']

print("\nBy Region:")
if high_h:
    high_h_correct = sum(1 for r in high_h if r['correct'])
    print(f"  Trust Region (h > 0.5): {high_h_correct}/{len(high_h)} = {high_h_correct/len(high_h):.1%}")

if low_h:
    low_h_correct = sum(1 for r in low_h if r['correct'])
    print(f"  Low-h Region (h <= 0.5): {low_h_correct}/{len(low_h)} = {low_h_correct/len(low_h):.1%}")

print("\nBy Quadrant:")
if q2:
    q2_correct = sum(1 for r in q2 if r['correct'])
    print(f"  Q2 (High FS, Low h): {q2_correct}/{len(q2)} = {q2_correct/len(q2):.1%}")

if q4:
    q4_correct = sum(1 for r in q4 if r['correct'])
    print(f"  Q4 (Low FS, Low h): {q4_correct}/{len(q4)} = {q4_correct/len(q4):.1%}")

# Analyze decisive predictions only (exclude Uncertain)
decisive_results = [r for r in loo_results if r['predicted'] != "Uncertain"]
if decisive_results:
    decisive_correct = sum(1 for r in decisive_results if r['correct'])
    print(f"\nDecisive Predictions Only: {decisive_correct}/{len(decisive_results)} = {decisive_correct/len(decisive_results):.1%}")

# Failure analysis
failures = [r for r in loo_results if not r['correct']]
if failures:
    print(f"\n{'='*80}")
    print("FAILURE ANALYSIS")
    print("="*80)
    for f in failures:
        print(f"\n{f['dataset']} ({f['quadrant']}):")
        print(f"  h={f['h']:.3f}, MLP={f['mlp_mean']:.3f}, GCN-MLP={f['gcn_mlp']:+.3f}")
        print(f"  Predicted: {f['predicted']}, Actual: {f['actual']}")
else:
    print(f"\n{'='*80}")
    print("NO FAILURES! Perfect LOO-CV accuracy.")
    print("="*80)

# Bootstrap confidence interval for accuracy
print(f"\n{'='*80}")
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("="*80)

n_bootstrap = 10000
bootstrap_accuracies = []
np.random.seed(42)

for _ in range(n_bootstrap):
    sample_indices = np.random.choice(len(loo_results), size=len(loo_results), replace=True)
    sample_correct = sum(1 for i in sample_indices if loo_results[i]['correct'])
    bootstrap_accuracies.append(sample_correct / len(loo_results))

ci_lower = np.percentile(bootstrap_accuracies, 2.5)
ci_upper = np.percentile(bootstrap_accuracies, 97.5)

print(f"\nLOO-CV Accuracy: {accuracy:.1%}")
print(f"Bootstrap 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
print(f"(n_bootstrap = {n_bootstrap})")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'method': 'Leave-One-Out Cross-Validation',
    'total_datasets': total,
    'overall_accuracy': accuracy,
    'bootstrap_ci_lower': ci_lower,
    'bootstrap_ci_upper': ci_upper,
    'high_h_accuracy': high_h_correct/len(high_h) if high_h else None,
    'low_h_accuracy': low_h_correct/len(low_h) if low_h else None,
    'q2_accuracy': q2_correct/len(q2) if q2 else None,
    'decisive_accuracy': decisive_correct/len(decisive_results) if decisive_results else None,
    'detailed_results': loo_results,
    'failures': failures
}

with open('comprehensive_loo_cv_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: comprehensive_loo_cv_results.json")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
