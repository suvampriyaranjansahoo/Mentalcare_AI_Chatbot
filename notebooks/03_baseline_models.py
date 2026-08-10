"""
Phase 3 — Real ML Baselines

Builds and cross-validates TF-IDF based baselines on data/train.csv ONLY.
data/evaluation.csv is NOT touched in this phase (model selection only).

Models:
  1. TF-IDF + Logistic Regression
  2. TF-IDF + Linear SVM
  3. TF-IDF + Multinomial Naive Bayes

Evaluation: 5-fold stratified cross-validation on the training set, reporting
accuracy, macro F1, weighted F1 (mean +/- std across folds).

Saves: evaluation/baseline_results.json

Run: python notebooks/03_baseline_models.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "baseline_results.json")

RANDOM_SEED = 42
N_FOLDS = 5

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
X = train_df["user_input"].astype(str)
y = train_df["emotion_label"]

models = {
    "TF-IDF + Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ]),
    "TF-IDF + Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(random_state=RANDOM_SEED)),
    ]),
    "TF-IDF + Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB()),
    ]),
}

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

results = {}

for name, pipeline in models.items():
    print(f"Running {N_FOLDS}-fold CV for: {name}")
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    model_result = {
        "cv_folds": N_FOLDS,
        "training_approach": f"{N_FOLDS}-fold stratified cross-validation on train.csv (n={len(train_df)})",
    }
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        model_result[metric] = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "fold_scores": [round(float(s), 4) for s in scores],
        }
    results[name] = model_result
    print(
        f"  accuracy={model_result['accuracy']['mean']:.4f} (+/-{model_result['accuracy']['std']:.4f})  "
        f"macro_f1={model_result['macro_f1']['mean']:.4f} (+/-{model_result['macro_f1']['std']:.4f})  "
        f"weighted_f1={model_result['weighted_f1']['mean']:.4f} (+/-{model_result['weighted_f1']['std']:.4f})"
    )

metadata = {
    "dataset": "data/train.csv",
    "train_size": len(train_df),
    "random_seed": RANDOM_SEED,
    "cv_strategy": f"StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_SEED})",
    "note": "Held-out data/evaluation.csv was NOT used at any point in this phase.",
}

output = {"metadata": metadata, "results": results}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {RESULTS_PATH}")

# ---------------------------------------------------------------
# Print a comparison table
# ---------------------------------------------------------------
print("\n=== Model Comparison (5-fold CV on train.csv) ===")
print(f"{'Model':<35} {'Accuracy':<18} {'Macro F1':<18} {'Weighted F1':<18}")
for name, r in results.items():
    acc = f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['std']:.4f}"
    mf1 = f"{r['macro_f1']['mean']:.4f} ± {r['macro_f1']['std']:.4f}"
    wf1 = f"{r['weighted_f1']['mean']:.4f} ± {r['weighted_f1']['std']:.4f}"
    print(f"{name:<35} {acc:<18} {mf1:<18} {wf1:<18}")
