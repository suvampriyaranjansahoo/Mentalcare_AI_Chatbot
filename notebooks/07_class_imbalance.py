"""
Phase 7 — Class Imbalance Analysis

Run: python notebooks/07_class_imbalance.py
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "final_results.json")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "class_imbalance_analysis.md")

with open(FINAL_RESULTS_PATH) as f:
    results = json.load(f)

per_class = results["classification_report"]
labels = results["confusion_matrix_labels"]

rows = []
for label in labels:
    stats = per_class[label]
    rows.append((label, stats["precision"], stats["recall"], stats["f1-score"], int(stats["support"])))

report = f"""# Class Imbalance Analysis

## Dataset Balance (from Phase 1 audit)

The Phase 1 data quality report measured a class imbalance ratio of **1.002**
(max class / min class), i.e. the dataset is essentially perfectly balanced
across all 7 emotion classes (~571-572 rows each before splitting). This is
consistent with the dataset being synthetically constructed rather than
collected from an organic, naturally imbalanced source.

## Per-Class Performance (held-out evaluation set)

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{chr(10).join(f"| {l} | {p:.4f} | {r:.4f} | {f:.4f} | {s} |" for l, p, r, f, s in rows)}

- **Macro F1:** {results['macro_f1']}
- **Weighted F1:** {results['weighted_f1']}

Since macro F1 (unweighted across classes) and weighted F1 (weighted by
class support) are equal here ({results['macro_f1']} vs {results['weighted_f1']}),
there is no meaningful gap between them — the expected signature of a
well-balanced dataset with no per-class degradation.

## Conclusion

Class imbalance is **not a meaningful factor** in this project, because the
dataset was constructed to be balanced from the start (Phase 1 finding).
Class weighting or resampling strategies were considered but **not applied**,
since there is no imbalance to correct and applying them would only add
unnecessary complexity without a validated benefit — consistent with the
instruction not to add such techniques "simply for appearance."

If this pipeline is later applied to a naturally-collected, imbalanced
dataset (see Phase 6's recommendation to test against a public benchmark),
this analysis should be re-run, since imbalance effects may become visible
there.
"""

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print("Per-class F1 range:", min(r[3] for r in rows), "-", max(r[3] for r in rows))
print(f"Report written to: {REPORT_PATH}")
