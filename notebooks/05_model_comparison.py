"""
Phase 5 — Model Comparison & Final Evaluation

Builds the comparison table from real results:
  - Phase 3 baseline CV results (evaluation/baseline_results.json)
  - Phase 4 transformer results (evaluation/transformer_results.json), IF that
    file exists — since Phase 4 must be run separately (see its docstring),
    this script works correctly whether or not it's been run yet.

Selects a final model based on validation (CV) performance, then runs that
ONE model ONE time on the held-out evaluation.csv to produce the final
reported metric. Saves the fitted model + vectorizer to artifacts/models/.

Run: python notebooks/05_model_comparison.py
"""

import os
import json
import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
EVAL_PATH = os.path.join(BASE_DIR, "data", "evaluation.csv")
BASELINE_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "baseline_results.json")
TRANSFORMER_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "transformer_results.json")
FINAL_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "final_results.json")
COMPARISON_MD_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.md")
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "models", "final_model.joblib")

RANDOM_SEED = 42

os.makedirs(os.path.dirname(FINAL_RESULTS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_ARTIFACT_PATH), exist_ok=True)


def load_baseline_results():
    with open(BASELINE_RESULTS_PATH) as f:
        return json.load(f)["results"]


def load_transformer_results():
    if not os.path.exists(TRANSFORMER_RESULTS_PATH):
        return None
    with open(TRANSFORMER_RESULTS_PATH) as f:
        return json.load(f)


def build_comparison_table(baseline_results, transformer_results):
    rows = []
    for name, r in baseline_results.items():
        rows.append({
            "model": name,
            "accuracy": r["accuracy"]["mean"],
            "accuracy_std": r["accuracy"]["std"],
            "macro_f1": r["macro_f1"]["mean"],
            "macro_f1_std": r["macro_f1"]["std"],
            "weighted_f1": r["weighted_f1"]["mean"],
            "weighted_f1_std": r["weighted_f1"]["std"],
            "training_approach": r["training_approach"],
            "status": "evaluated (5-fold CV on train.csv)",
        })

    if transformer_results is not None:
        ft = transformer_results["fine_tuned"]["final_held_out_metrics"]
        rows.append({
            "model": f"Fine-tuned {transformer_results['model_name']}",
            "accuracy": ft["accuracy"],
            "accuracy_std": None,
            "macro_f1": ft["macro_f1"],
            "macro_f1_std": None,
            "weighted_f1": ft["weighted_f1"],
            "weighted_f1_std": None,
            "training_approach": "Fine-tuned on train.csv, evaluated once on held-out evaluation.csv",
            "status": "evaluated (held-out set, single run)",
        })
    else:
        rows.append({
            "model": "Fine-tuned j-hartmann/emotion-english-distilroberta-base",
            "accuracy": None,
            "accuracy_std": None,
            "macro_f1": None,
            "macro_f1_std": None,
            "weighted_f1": None,
            "weighted_f1_std": None,
            "training_approach": "Not yet run in this environment (requires huggingface.co access — "
                                  "see notebooks/04_transformer_training.py to run locally)",
            "status": "PENDING — script provided, not executed",
        })

    return rows


def select_final_model(baseline_results):
    """
    Selection criterion: highest mean weighted F1 in cross-validation on
    train.csv. Ties broken by lower std (more stable across folds).
    This mirrors what would be done even if the transformer were available:
    selection happens on validation performance, never on the held-out set.
    """
    ranked = sorted(
        baseline_results.items(),
        key=lambda kv: (-kv[1]["weighted_f1"]["mean"], kv[1]["weighted_f1"]["std"]),
    )
    return ranked[0]


def main():
    baseline_results = load_baseline_results()
    transformer_results = load_transformer_results()

    comparison_rows = build_comparison_table(baseline_results, transformer_results)

    # ---- Selection ----
    selected_name, selected_stats = select_final_model(baseline_results)
    selection_reason = (
        f"'{selected_name}' selected: highest mean weighted F1 in 5-fold CV on train.csv "
        f"({selected_stats['weighted_f1']['mean']:.4f} ± {selected_stats['weighted_f1']['std']:.4f}). "
        f"Transformer fine-tuning (Phase 4) was not executable in this environment "
        f"(no huggingface.co access); once run locally, its held-out result should be "
        f"compared against this baseline before final selection is treated as permanent."
    )
    print(selection_reason)

    # ---- Train the selected model on FULL train.csv, evaluate ONCE on evaluation.csv ----
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    pipeline_components = {
        "TF-IDF + Logistic Regression": lambda: LinearSVC(random_state=RANDOM_SEED),  # placeholder, replaced below
    }

    # Rebuild the actual chosen pipeline explicitly based on name
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    if "SVM" in selected_name:
        clf = LinearSVC(random_state=RANDOM_SEED)
    elif "Logistic" in selected_name:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    else:
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()

    X_train = vectorizer.fit_transform(train_df["user_input"].astype(str))
    y_train = train_df["emotion_label"]
    clf.fit(X_train, y_train)

    X_eval = vectorizer.transform(eval_df["user_input"].astype(str))
    y_eval = eval_df["emotion_label"]
    y_pred = clf.predict(X_eval)

    labels_sorted = sorted(y_train.unique())

    final_metrics = {
        "model": selected_name,
        "dataset_size": len(train_df) + len(eval_df),
        "evaluation_size": len(eval_df),
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_eval, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_eval, y_pred, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_eval, y_pred, average="macro")), 4),
        "macro_recall": round(float(recall_score(y_eval, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=labels_sorted).tolist(),
        "confusion_matrix_labels": labels_sorted,
        "classification_report": classification_report(
            y_eval, y_pred, labels=labels_sorted, output_dict=True, zero_division=0
        ),
        "selection_reason": selection_reason,
        "random_seed": RANDOM_SEED,
    }

    with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nFinal held-out metrics saved to: {FINAL_RESULTS_PATH}")
    print(f"Accuracy: {final_metrics['accuracy']}, Macro F1: {final_metrics['macro_f1']}, Weighted F1: {final_metrics['weighted_f1']}")

    # ---- Save the fitted artifact for the Flask app to load ----
    joblib.dump({"vectorizer": vectorizer, "model": clf, "labels": labels_sorted}, MODEL_ARTIFACT_PATH)
    print(f"Model artifact saved to: {MODEL_ARTIFACT_PATH}")

    # ---- Write comparison markdown ----
    md_lines = [
        "# Model Comparison\n",
        "All cross-validation numbers below come from `evaluation/baseline_results.json` "
        "(5-fold stratified CV on `data/train.csv`). The transformer row is populated only "
        "once `notebooks/04_transformer_training.py` has been run and its output copied into "
        "`evaluation/transformer_results.json`.\n",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Approach | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in comparison_rows:
        acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "—"
        if row["accuracy_std"]:
            acc += f" ± {row['accuracy_std']:.4f}"
        mf1 = f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "—"
        if row["macro_f1_std"]:
            mf1 += f" ± {row['macro_f1_std']:.4f}"
        wf1 = f"{row['weighted_f1']:.4f}" if row["weighted_f1"] is not None else "—"
        if row["weighted_f1_std"]:
            wf1 += f" ± {row['weighted_f1_std']:.4f}"
        md_lines.append(f"| {row['model']} | {acc} | {mf1} | {wf1} | {row['training_approach']} | {row['status']} |")

    md_lines += [
        "",
        f"## Selection",
        "",
        selection_reason,
        "",
        f"## Final Held-Out Evaluation (evaluation.csv, n={final_metrics['evaluation_size']})",
        "",
        f"- **Accuracy:** {final_metrics['accuracy']}",
        f"- **Macro F1:** {final_metrics['macro_f1']}",
        f"- **Weighted F1:** {final_metrics['weighted_f1']}",
        "",
        "⚠️ **Important caveat**, carried over from the Phase 1 data audit: this dataset's "
        "vocabulary is only 210 unique tokens across 4,000 rows, generated from a small set "
        "of sentence templates. Both the CV and held-out numbers above are near-ceiling "
        "(99–100%) as a direct consequence of that — they demonstrate the pipeline and "
        "methodology work correctly, but should **not** be read as evidence of strong "
        "generalization to naturalistic, diverse real-world text. See `LIMITATIONS.md`.",
    ]

    os.makedirs(os.path.dirname(COMPARISON_MD_PATH), exist_ok=True)
    with open(COMPARISON_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Comparison table written to: {COMPARISON_MD_PATH}")


if __name__ == "__main__":
    main()
