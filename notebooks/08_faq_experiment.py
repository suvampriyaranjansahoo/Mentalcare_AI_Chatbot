"""
Phase 8 — Fuzzy FAQ Retrieval Experiment

Builds a FAQ validation set from held-out data and evaluates several
similarity approaches for matching a new user message against the FAQ
bank (train.csv), determining an evidence-based confidence threshold
instead of an arbitrary one.

Approaches compared:
  1. difflib.SequenceMatcher (character-level similarity — what the
     original project used via get_close_matches)
  2. Token-set Jaccard similarity
  3. TF-IDF cosine similarity

For each approach, we sweep thresholds and measure:
  - match accuracy (retrieved FAQ's label matches the query's true label)
  - false-match rate (confidently wrong matches)
  - unmatched rate (nothing cleared the threshold)

The FAQ bank is train.csv. The validation queries are evaluation.csv
(held out, never used for threshold tuning in earlier phases — used here
specifically for FAQ threshold selection, which is a different task from
the classifier evaluation in Phase 5).

Run: python notebooks/08_faq_experiment.py
"""

import os
import json
import difflib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
EVAL_PATH = os.path.join(BASE_DIR, "data", "evaluation.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "faq_results.json")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "faq_matching_report.md")
PLOTS_DIR = os.path.join(BASE_DIR, "reports", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
eval_df = pd.read_csv(EVAL_PATH).reset_index(drop=True)

faq_texts = train_df["user_input"].astype(str).tolist()
faq_labels = train_df["emotion_label"].tolist()

query_texts = eval_df["user_input"].astype(str).tolist()
query_labels = eval_df["emotion_label"].tolist()

THRESHOLDS = np.arange(0.3, 1.0, 0.05)


def evaluate_at_threshold(sim_matrix, threshold):
    """sim_matrix: (n_queries, n_faq). Returns match_accuracy, false_match_rate, unmatched_rate."""
    n = sim_matrix.shape[0]
    best_idx = np.argmax(sim_matrix, axis=1)
    best_sim = sim_matrix[np.arange(n), best_idx]

    matched_mask = best_sim >= threshold
    n_matched = matched_mask.sum()
    n_unmatched = n - n_matched

    if n_matched == 0:
        return {"match_accuracy": None, "false_match_rate": None, "unmatched_rate": 1.0, "n_matched": 0}

    matched_true = np.array(query_labels)[matched_mask]
    matched_pred = np.array(faq_labels)[best_idx[matched_mask]]
    correct = (matched_true == matched_pred).sum()

    return {
        "match_accuracy": round(float(correct / n_matched), 4),
        "false_match_rate": round(float((n_matched - correct) / n), 4),
        "unmatched_rate": round(float(n_unmatched / n), 4),
        "n_matched": int(n_matched),
    }


# ---------------------------------------------------------------
# Approach 1: difflib character similarity
# ---------------------------------------------------------------
def difflib_sim_matrix():
    n_q, n_f = len(query_texts), len(faq_texts)
    sim = np.zeros((n_q, n_f))
    for i, q in enumerate(query_texts):
        for j, f in enumerate(faq_texts):
            sim[i, j] = difflib.SequenceMatcher(None, q.lower(), f.lower()).ratio()
    return sim


# ---------------------------------------------------------------
# Approach 2: Token-set Jaccard similarity
# ---------------------------------------------------------------
def jaccard_sim_matrix():
    n_q, n_f = len(query_texts), len(faq_texts)
    sim = np.zeros((n_q, n_f))
    faq_token_sets = [set(f.lower().split()) for f in faq_texts]
    for i, q in enumerate(query_texts):
        q_tokens = set(q.lower().split())
        for j, f_tokens in enumerate(faq_token_sets):
            union = q_tokens | f_tokens
            sim[i, j] = len(q_tokens & f_tokens) / len(union) if union else 0.0
    return sim


# ---------------------------------------------------------------
# Approach 3: TF-IDF cosine similarity
# ---------------------------------------------------------------
def tfidf_sim_matrix():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    faq_vecs = vectorizer.fit_transform(faq_texts)
    query_vecs = vectorizer.transform(query_texts)
    return cosine_similarity(query_vecs, faq_vecs)


print("Computing similarity matrices (this may take a moment for difflib)...")
approaches = {
    "difflib_character_similarity": difflib_sim_matrix(),
    "token_jaccard_similarity": jaccard_sim_matrix(),
    "tfidf_cosine_similarity": tfidf_sim_matrix(),
}

results = {}
for name, sim_matrix in approaches.items():
    print(f"Sweeping thresholds for: {name}")
    threshold_results = {}
    for t in THRESHOLDS:
        threshold_results[round(float(t), 2)] = evaluate_at_threshold(sim_matrix, t)
    results[name] = {
        "threshold_sweep": threshold_results,
        "similarity_score_distribution": {
            "mean_best_score": round(float(np.mean(np.max(sim_matrix, axis=1))), 4),
            "std_best_score": round(float(np.std(np.max(sim_matrix, axis=1))), 4),
            "min_best_score": round(float(np.min(np.max(sim_matrix, axis=1))), 4),
            "max_best_score": round(float(np.max(np.max(sim_matrix, axis=1))), 4),
        },
    }

# ---------------------------------------------------------------
# Determine evidence-based threshold per approach:
# highest threshold where unmatched_rate <= 0.5 AND match_accuracy is maximized
# (i.e. best accuracy among thresholds that still match at least half of queries)
# ---------------------------------------------------------------
def pick_best_threshold(threshold_results):
    candidates = [
        (t, r) for t, r in threshold_results.items()
        if r["match_accuracy"] is not None and r["unmatched_rate"] <= 0.5
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda tr: (-tr[1]["match_accuracy"], tr[1]["unmatched_rate"]))
    return candidates[0]


chosen_thresholds = {}
for name, r in results.items():
    best = pick_best_threshold(r["threshold_sweep"])
    chosen_thresholds[name] = best

# ---------------------------------------------------------------
# Plot: accuracy vs threshold per approach
# ---------------------------------------------------------------
plt.figure(figsize=(9, 6))
for name, r in results.items():
    ts = sorted(r["threshold_sweep"].keys())
    accs = [r["threshold_sweep"][t]["match_accuracy"] or 0 for t in ts]
    plt.plot(ts, accs, marker="o", label=name)
plt.xlabel("Similarity threshold")
plt.ylabel("Match accuracy (on matched queries)")
plt.title("FAQ Match Accuracy vs. Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "faq_threshold_sweep.png"), dpi=150)
plt.close()

output = {
    "faq_bank_size": len(faq_texts),
    "query_set_size": len(query_texts),
    "approaches": results,
    "chosen_thresholds": {
        name: {"threshold": t, "metrics": m} if t else None
        for name, (t, m) in ((n, (b[0], b[1]) if b else (None, None)) for n, b in chosen_thresholds.items())
    },
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------
md_lines = [
    "# FAQ Fuzzy Matching Experiment\n",
    f"FAQ bank: `train.csv` ({len(faq_texts)} entries). Validation queries: `evaluation.csv` ({len(query_texts)} entries).\n",
    "## Similarity Score Distributions\n",
    "| Approach | Mean best-match score | Std | Min | Max |",
    "|---|---|---|---|---|",
]
for name, r in results.items():
    d = r["similarity_score_distribution"]
    md_lines.append(f"| {name} | {d['mean_best_score']} | {d['std_best_score']} | {d['min_best_score']} | {d['max_best_score']} |")

md_lines += [
    "",
    "## Evidence-Based Threshold Selection\n",
    "Selection rule: among thresholds where at least 50% of queries still get "
    "matched, pick the threshold with the highest match accuracy (not an "
    "arbitrary fixed value like 0.80).\n",
    "| Approach | Chosen threshold | Match accuracy | False-match rate | Unmatched rate |",
    "|---|---|---|---|---|",
]
for name, (t, m) in chosen_thresholds.items():
    if t is None:
        md_lines.append(f"| {name} | — | no threshold kept ≥50% match rate | — | — |")
    else:
        md_lines.append(f"| {name} | {t} | {m['match_accuracy']} | {m['false_match_rate']} | {m['unmatched_rate']} |")

md_lines += [
    "",
    "![Threshold Sweep](plots/faq_threshold_sweep.png)",
    "",
    "## Interpretation\n",
    "Because this dataset is templated with limited vocabulary (see Phase 1 "
    "finding: 210 unique tokens), all three similarity approaches likely "
    "perform very well at matching queries back to their originating template "
    "family, in the same way the classifier baselines did. This is an honest "
    "measurement, but the same caveat applies as in the model comparison: "
    "these numbers reflect performance on templated text, not necessarily on "
    "naturalistic free-form user input the way the original project's live "
    "chatbot would receive it in production.",
    "",
    "The original project used `difflib.get_close_matches` with a fixed "
    "`cutoff=0.5`. Comparing that fixed value against the evidence-based "
    "thresholds chosen above shows whether 0.5 was a reasonable choice or "
    "arbitrary luck.",
]

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("\nChosen thresholds:")
for name, (t, m) in chosen_thresholds.items():
    print(f"  {name}: threshold={t}, metrics={m}")
print(f"\nResults saved to: {RESULTS_PATH}")
print(f"Report saved to: {REPORT_PATH}")
