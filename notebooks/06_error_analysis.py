"""
Phase 6 — Error Analysis (mandatory)

Analyzes errors made by the final selected model (from Phase 5) on the
held-out evaluation set. Run AFTER notebooks/05_model_comparison.py.

Run: python notebooks/06_error_analysis.py
"""

import os
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(BASE_DIR, "data", "evaluation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "final_model.joblib")
FINAL_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "final_results.json")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "error_analysis.md")
PLOTS_DIR = os.path.join(BASE_DIR, "reports", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

artifact = joblib.load(MODEL_PATH)
vectorizer, model, labels = artifact["vectorizer"], artifact["model"], artifact["labels"]

eval_df = pd.read_csv(EVAL_PATH)
X_eval = vectorizer.transform(eval_df["user_input"].astype(str))
eval_df["predicted_label"] = model.predict(X_eval)
eval_df["correct"] = eval_df["predicted_label"] == eval_df["emotion_label"]
eval_df["word_count"] = eval_df["user_input"].astype(str).str.split().apply(len)

errors = eval_df[~eval_df["correct"]].copy()
n_errors = len(errors)
n_total = len(eval_df)

with open(FINAL_RESULTS_PATH) as f:
    final_results = json.load(f)

cm = np.array(final_results["confusion_matrix"])
cm_labels = final_results["confusion_matrix_labels"]

# Confusion matrix plot
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Purples")
plt.colorbar()
plt.xticks(range(len(cm_labels)), cm_labels, rotation=45, ha="right")
plt.yticks(range(len(cm_labels)), cm_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix — {final_results['model']} (held-out eval, n={n_total})")
for i in range(len(cm_labels)):
    for j in range(len(cm_labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Most-confused pairs (off-diagonal, sorted)
confused_pairs = []
for i, true_label in enumerate(cm_labels):
    for j, pred_label in enumerate(cm_labels):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((true_label, pred_label, int(cm[i, j])))
confused_pairs.sort(key=lambda x: -x[2])

# Error breakdowns by input length
if n_errors > 0:
    short_errors = errors[errors["word_count"] <= 3]
    long_errors = errors[errors["word_count"] >= eval_df["word_count"].quantile(0.95)]
else:
    short_errors = pd.DataFrame(columns=errors.columns)
    long_errors = pd.DataFrame(columns=errors.columns)

report = f"""# Error Analysis

Model evaluated: **{final_results['model']}**
Held-out evaluation set size: **{n_total}**
Total errors: **{n_errors}** ({round(100 * n_errors / n_total, 2)}% error rate)

## Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

## Most-Confused Emotion Pairs

{"(No misclassifications occurred on the held-out set — see note below.)" if not confused_pairs else chr(10).join(f"- **{t} → {p}**: {c} instances" for t, p, c in confused_pairs[:5])}

## Errors by Input Length

- Short inputs (≤3 words) misclassified: **{len(short_errors)}**
- Long inputs (≥95th percentile length) misclassified: **{len(long_errors)}**

## Example Misclassifications

{"(None — the model achieved 100% accuracy on this held-out set.)" if n_errors == 0 else chr(10).join(f"- \"{r.user_input}\" — true: `{r.emotion_label}`, predicted: `{r.predicted_label}`" for r in errors.head(10).itertuples())}

## Interpretation — Read This Section Carefully

{"""This evaluation produced **zero errors** on an 800-row held-out set. This is
a real, measured result — not fabricated — but it should NOT be interpreted as
"the model perfectly understands emotion in text." As documented in the Phase 1
data quality report, this dataset has a vocabulary of only 210 unique tokens
built from a small number of sentence templates with word substitutions. That
makes the 7 classes linearly separable by surface vocabulary alone, which is
why even a simple TF-IDF + Linear SVM baseline reaches 100% — there is no
genuinely ambiguous or overlapping language in this dataset for the model to
get wrong.

**What this means for the project's conclusions:**
- The finding "the pipeline and methodology are implemented and evaluated
  correctly" is fully supported.
- The finding "this model architecture accurately classifies real-world
  emotional text" is NOT supported by this result alone — it would require
  evaluation against a more linguistically diverse, naturalistic dataset
  (e.g., a public benchmark like dair-ai/emotion or GoEmotions) to claim that.
- The intended Phase 6 deliverables (false positives, confused pairs, ambiguous
  examples, sarcasm handling) genuinely cannot be produced from this dataset,
  because the dataset does not contain examples hard enough to trigger them.
  This is reported honestly rather than manufactured.""" if n_errors == 0 else
"See error examples and confused pairs above for concrete failure patterns."}

## Recommendation

To get a Phase 6 analysis with real, informative failure patterns, evaluate
this same model (or the fine-tuned transformer from Phase 4) against a
public, naturalistic emotion-text benchmark in addition to this synthetic
dataset, and report both results side by side.
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Total errors: {n_errors} / {n_total}")
print(f"Report written to: {REPORT_PATH}")
