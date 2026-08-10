import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

EXPECTED_LABELS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}


def test_raw_data_loads_and_has_expected_columns():
    df = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))
    assert set(df.columns) == {"user_input", "emotion_label", "bot_response"}
    assert len(df) > 0


def test_raw_data_has_no_missing_values():
    df = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))
    assert df.isnull().sum().sum() == 0


def test_raw_data_labels_match_expected_set():
    df = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))
    assert set(df["emotion_label"].unique()) == EXPECTED_LABELS


def test_train_eval_split_exists_and_no_overlap():
    train_path = os.path.join(DATA_DIR, "train.csv")
    eval_path = os.path.join(DATA_DIR, "evaluation.csv")
    assert os.path.exists(train_path), "Run notebooks/02_data_split.py first"
    assert os.path.exists(eval_path), "Run notebooks/02_data_split.py first"

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    train_texts = set(train_df["user_input"].str.lower().str.strip())
    eval_texts = set(eval_df["user_input"].str.lower().str.strip())
    overlap = train_texts & eval_texts
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and eval"


def test_split_is_stratified_reasonably():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    eval_df = pd.read_csv(os.path.join(DATA_DIR, "evaluation.csv"))

    train_dist = train_df["emotion_label"].value_counts(normalize=True)
    eval_dist = eval_df["emotion_label"].value_counts(normalize=True)

    for label in EXPECTED_LABELS:
        # Proportions between train/eval should be reasonably close (within 5 percentage points)
        assert abs(train_dist.get(label, 0) - eval_dist.get(label, 0)) < 0.05
