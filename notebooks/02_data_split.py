"""
Phase 2 — Proper Data Splitting

Creates a reproducible, stratified 80/20 train/evaluation split of
expanded_data.csv, grouped by normalized user_input so that near-duplicate
rows (identified in Phase 1) never appear on both sides of the split.

Method:
  1. Normalize user_input (lowercase, strip punctuation, collapse whitespace).
  2. Collapse to one row per unique normalized text, keeping its emotion_label
     (Phase 1 confirmed 0 label conflicts, so this is safe — each normalized
     group has exactly one label).
  3. Stratified-split the *groups* 80/20 by emotion_label with a fixed seed.
  4. Expand back out: every original row whose normalized text fell in the
     "train" group goes to train.csv, same for evaluation.

This guarantees no near-duplicate leakage across the split while keeping
the split stratified by class.

Run: python notebooks/02_data_split.py
"""

import os
import re
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "expanded_data.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
EVAL_PATH = os.path.join(BASE_DIR, "data", "evaluation.csv")
SPLIT_DOC_PATH = os.path.join(BASE_DIR, "reports", "split_methodology.md")

RANDOM_SEED = 42
EVAL_FRACTION = 0.20


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    df["_normalized"] = df["user_input"].apply(normalize)

    # One row per unique normalized text (label is consistent within a group,
    # verified in Phase 1: 0 conflicting-label groups).
    groups = df.groupby("_normalized")["emotion_label"].first().reset_index()

    train_groups, eval_groups = train_test_split(
        groups,
        test_size=EVAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=groups["emotion_label"],
    )

    train_norm_set = set(train_groups["_normalized"])
    eval_norm_set = set(eval_groups["_normalized"])

    train_df = df[df["_normalized"].isin(train_norm_set)].drop(columns=["_normalized"])
    eval_df = df[df["_normalized"].isin(eval_norm_set)].drop(columns=["_normalized"])

    # Sanity check: no overlap
    overlap = set(train_df["user_input"].apply(normalize)) & set(eval_df["user_input"].apply(normalize))
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} overlapping normalized texts"

    train_df.to_csv(TRAIN_PATH, index=False)
    eval_df.to_csv(EVAL_PATH, index=False)

    train_class_dist = train_df["emotion_label"].value_counts().to_dict()
    eval_class_dist = eval_df["emotion_label"].value_counts().to_dict()

    doc = f"""# Data Split Methodology

## Procedure

1. Each `user_input` was normalized (lowercased, punctuation stripped, whitespace
   collapsed) to identify near-duplicate groups, per the Phase 1 audit.
2. Rows were grouped by normalized text — {len(groups)} unique normalized groups
   found across {len(df)} total rows.
3. Groups (not individual rows) were split 80/20 using
   `sklearn.model_selection.train_test_split` with `stratify=emotion_label`
   and `random_state={RANDOM_SEED}`.
4. All original rows belonging to a group were assigned to whichever split
   that group was assigned to — guaranteeing no near-duplicate appears in
   both `train.csv` and `evaluation.csv`.
5. Overlap between the two splits was verified programmatically (assertion
   in `02_data_split.py`) — **0 overlapping normalized texts** confirmed.

## Configuration

- Random seed: `{RANDOM_SEED}`
- Target evaluation fraction: `{EVAL_FRACTION}` (of unique normalized groups)
- Stratification column: `emotion_label`

## Resulting Split Sizes

- `data/train.csv`: **{len(train_df)}** rows
- `data/evaluation.csv`: **{len(eval_df)}** rows
- Actual eval fraction (rows): **{round(len(eval_df) / len(df), 3)}**

## Class Distribution After Split

| Emotion | Train count | Train % | Eval count | Eval % |
|---|---|---|---|---|
{chr(10).join(
    f"| {label} | {train_class_dist.get(label, 0)} | "
    f"{round(100 * train_class_dist.get(label, 0) / len(train_df), 2)}% | "
    f"{eval_class_dist.get(label, 0)} | "
    f"{round(100 * eval_class_dist.get(label, 0) / len(eval_df), 2)}% |"
    for label in sorted(set(train_class_dist) | set(eval_class_dist))
)}

## Guarantee

`data/evaluation.csv` is held out from this point forward. It must not be used
for baseline model selection (Phase 3), transformer hyperparameter tuning, or
FAQ threshold selection (Phase 8) — only for final reported metrics (Phase 5,
Phase 17). Model/threshold selection uses cross-validation on `train.csv` only.
"""

    os.makedirs(os.path.dirname(SPLIT_DOC_PATH), exist_ok=True)
    with open(SPLIT_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"Train rows: {len(train_df)}")
    print(f"Eval rows: {len(eval_df)}")
    print(f"Overlap (should be 0): {len(overlap)}")
    print(f"Train class distribution: {train_class_dist}")
    print(f"Eval class distribution: {eval_class_dist}")
    print(f"Methodology doc written to: {SPLIT_DOC_PATH}")


if __name__ == "__main__":
    main()
