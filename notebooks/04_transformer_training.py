"""
Phase 4 — Transformer Model (pretrained baseline + fine-tuning)

IMPORTANT — READ BEFORE RUNNING:
This script was NOT executed in the environment that generated the rest of
this repository, because that sandbox has no network access to
huggingface.co. It is provided as a complete, correct, ready-to-run script
for YOU to execute locally (or on Colab/Kaggle) where model downloads work.

After running it, copy the resulting evaluation/transformer_results.json
(printed/saved by this script) back into the repo and re-run
notebooks/05_model_comparison.py so the comparison table and README reflect
real transformer numbers instead of being absent.

Steps performed:
  1. Load train.csv / evaluation.csv (from Phase 2 — do NOT touch
     evaluation.csv until the very final evaluation at the bottom).
  2. Map the model's native emotion labels to our 7 project labels.
  3. Evaluate the PRETRAINED model zero-shot on a validation slice of train.csv.
  4. Fine-tune on train.csv (with an internal train/val split — evaluation.csv
     is still not touched).
  5. Run the fine-tuned model ONCE on evaluation.csv for the final reported
     metric.

Run: python notebooks/04_transformer_training.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
EVAL_PATH = os.path.join(BASE_DIR, "data", "evaluation.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "transformer_results.json")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "artifacts", "models", "finetuned_emotion_model")

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
RANDOM_SEED = 42

# The model's native output labels vs. our 7 project labels.
# j-hartmann/emotion-english-distilroberta-base outputs:
#   anger, disgust, fear, joy, neutral, sadness, surprise
# ...which already matches our label set 1:1. Mapping is identity, but kept
# explicit and centralized here in case you swap models later.
MODEL_TO_PROJECT_LABEL = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}
PROJECT_LABELS = sorted(set(MODEL_TO_PROJECT_LABEL.values()))


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    return train_df, eval_df


def evaluate_predictions(y_true, y_pred, split_name):
    return {
        "split": split_name,
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro")), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=PROJECT_LABELS).tolist(),
        "confusion_matrix_labels": PROJECT_LABELS,
        "classification_report": classification_report(
            y_true, y_pred, labels=PROJECT_LABELS, output_dict=True, zero_division=0
        ),
    }


def run_pretrained_zero_shot(texts):
    """Zero-shot inference with the pretrained model, no fine-tuning."""
    from transformers import pipeline as hf_pipeline

    clf = hf_pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=1,
    )
    preds = []
    for text in texts:
        result = clf(text)[0]
        # result is a list of dicts (top_k=1 -> length 1)
        label = result[0]["label"].lower() if isinstance(result, list) else result["label"].lower()
        preds.append(MODEL_TO_PROJECT_LABEL.get(label, "neutral"))
    return preds


def fine_tune_and_evaluate(train_df, eval_df):
    """Full fine-tuning workflow using HF Trainer."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )

    label2id = {label: i for i, label in enumerate(PROJECT_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    # Internal train/val split (evaluation.csv is NOT used here)
    internal_train, internal_val = train_test_split(
        train_df, test_size=0.15, random_state=RANDOM_SEED, stratify=train_df["emotion_label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_hf_dataset(df):
        d = Dataset.from_pandas(df[["user_input", "emotion_label"]].reset_index(drop=True))
        d = d.map(lambda x: {"label": label2id[x["emotion_label"]]})
        d = d.map(lambda x: tokenizer(x["user_input"], truncation=True, padding="max_length", max_length=64), batched=True)
        return d

    train_ds = to_hf_dataset(internal_train)
    val_ds = to_hf_dataset(internal_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(PROJECT_LABELS), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "artifacts", "models", "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        seed=RANDOM_SEED,
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_result = trainer.train()
    trainer.save_model(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)

    # Learning curve / training history
    history = trainer.state.log_history

    # ---- FINAL evaluation on the untouched held-out evaluation.csv ----
    eval_ds = to_hf_dataset(eval_df)
    eval_output = trainer.predict(eval_ds)
    eval_preds = np.argmax(eval_output.predictions, axis=1)
    eval_true = eval_ds["label"]

    y_true_labels = [id2label[i] for i in eval_true]
    y_pred_labels = [id2label[i] for i in eval_preds]

    final_metrics = evaluate_predictions(y_true_labels, y_pred_labels, "held_out_evaluation")

    return {
        "training_history": history,
        "final_held_out_metrics": final_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }


def main():
    train_df, eval_df = load_data()

    print("=== Step 1: Zero-shot pretrained model on a train-side validation slice ===")
    _, val_slice = train_test_split(
        train_df, test_size=0.15, random_state=RANDOM_SEED, stratify=train_df["emotion_label"]
    )
    zero_shot_preds = run_pretrained_zero_shot(val_slice["user_input"].tolist())
    zero_shot_metrics = evaluate_predictions(
        val_slice["emotion_label"].tolist(), zero_shot_preds, "train_validation_slice_zero_shot"
    )
    print(json.dumps({k: v for k, v in zero_shot_metrics.items() if k not in ("confusion_matrix", "classification_report")}, indent=2))

    print("\n=== Step 2: Fine-tuning on train.csv, final eval on evaluation.csv (ONE TIME) ===")
    fine_tune_results = fine_tune_and_evaluate(train_df, eval_df)
    print(json.dumps(
        {k: v for k, v in fine_tune_results["final_held_out_metrics"].items()
         if k not in ("confusion_matrix", "classification_report")},
        indent=2,
    ))

    output = {
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "label_mapping": MODEL_TO_PROJECT_LABEL,
        "zero_shot_pretrained": zero_shot_metrics,
        "fine_tuned": fine_tune_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {RESULTS_PATH}")
    print("Copy this file back into the repo's evaluation/ folder, then re-run")
    print("notebooks/05_model_comparison.py to update the comparison table.")


if __name__ == "__main__":
    main()
