"""Validate source data before splitting or training."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

EXPECTED_LABELS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", value.lower())).strip()


def validate(path: Path) -> dict:
    frame = pd.read_csv(path)
    required = {"user_input", "emotion_label", "bot_response"}
    if set(frame.columns) != required:
        raise ValueError(f"Expected columns: {sorted(required)}")
    blank = frame[list(required)].isna().any(axis=1) | frame[list(required)].astype(str).apply(lambda col: col.str.strip().eq("")).any(axis=1)
    normalized = frame["user_input"].astype(str).map(normalize)
    labels = set(frame["emotion_label"])
    conflict_count = int(pd.DataFrame({"text": normalized, "label": frame["emotion_label"]}).groupby("text").label.nunique().gt(1).sum())
    report = {"rows": len(frame), "missing_or_blank_rows": int(blank.sum()), "exact_duplicate_rows": int(frame.duplicated().sum()), "normalized_duplicate_rows": int(normalized.duplicated().sum()), "invalid_labels": sorted(labels - EXPECTED_LABELS), "missing_labels": sorted(EXPECTED_LABELS - labels), "conflicting_normalized_labels": conflict_count, "class_distribution": frame["emotion_label"].value_counts().to_dict()}
    if report["missing_or_blank_rows"] or report["invalid_labels"] or report["missing_labels"] or conflict_count:
        raise ValueError(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/expanded_data.csv")
    parser.add_argument("--output", default="reports/data_validation.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = validate(Path(args.input))
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
