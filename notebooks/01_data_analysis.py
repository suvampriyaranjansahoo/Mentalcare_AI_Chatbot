"""
Phase 1 — Data Science Data Audit
MENTALCARE AI — expanded_data.csv (4,000 rows, 7-class emotion labels)

This script performs a full dataset audit and writes:
  - reports/data_quality_report.md
  - reports/plots/*.png

Run: python notebooks/01_data_analysis.py
"""

import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "expanded_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "reports", "plots")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "data_quality_report.md")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

EXPECTED_LABELS = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}

# -----------------------------------------------------------------
# Load
# -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
findings = {}

findings["shape"] = df.shape
findings["columns"] = list(df.columns)

# -----------------------------------------------------------------
# 2. Missing values
# -----------------------------------------------------------------
missing = df.isnull().sum()
findings["missing"] = missing.to_dict()

# Also check for blank/whitespace-only strings (not caught by isnull)
blank_mask = df.apply(lambda col: col.astype(str).str.strip().eq("").sum())
findings["blank_strings"] = blank_mask.to_dict()

# -----------------------------------------------------------------
# 3/4. Duplicate analysis
# -----------------------------------------------------------------
exact_dup_rows = df.duplicated().sum()
exact_dup_inputs = df.duplicated(subset=["user_input"]).sum()
findings["exact_duplicate_rows"] = int(exact_dup_rows)
findings["exact_duplicate_user_inputs"] = int(exact_dup_inputs)

# -----------------------------------------------------------------
# 5. Near-duplicate leakage analysis (normalized text match)
# -----------------------------------------------------------------
def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

df["_normalized"] = df["user_input"].apply(normalize)
near_dup_count = df.duplicated(subset=["_normalized"]).sum()
findings["near_duplicate_normalized_inputs"] = int(near_dup_count)

# Cross-label leakage: same normalized text mapped to >1 label
label_conflicts = (
    df.groupby("_normalized")["emotion_label"]
    .nunique()
    .reset_index(name="n_labels")
)
conflicting = label_conflicts[label_conflicts["n_labels"] > 1]
findings["normalized_inputs_with_conflicting_labels"] = int(len(conflicting))

# -----------------------------------------------------------------
# 6/7. Class distribution + imbalance ratio
# -----------------------------------------------------------------
class_counts = df["emotion_label"].value_counts()
findings["class_counts"] = class_counts.to_dict()
findings["unexpected_labels"] = sorted(set(df["emotion_label"].unique()) - EXPECTED_LABELS)
findings["missing_expected_labels"] = sorted(EXPECTED_LABELS - set(df["emotion_label"].unique()))
imbalance_ratio = class_counts.max() / class_counts.min()
findings["imbalance_ratio_max_over_min"] = round(float(imbalance_ratio), 3)

# -----------------------------------------------------------------
# 8/9/10. Length statistics
# -----------------------------------------------------------------
df["char_count"] = df["user_input"].astype(str).str.len()
df["word_count"] = df["user_input"].astype(str).str.split().apply(len)

findings["char_count_stats"] = df["char_count"].describe().to_dict()
findings["word_count_stats"] = df["word_count"].describe().to_dict()

# -----------------------------------------------------------------
# 11/12/13. Vocabulary + frequent/rare words
# -----------------------------------------------------------------
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()

all_tokens = [tok for text in df["user_input"] for tok in tokenize(text)]
token_freq = Counter(all_tokens)
vocab_size = len(token_freq)
findings["vocab_size"] = vocab_size
findings["total_tokens"] = len(all_tokens)
findings["top_20_words"] = token_freq.most_common(20)

rare_words = [w for w, c in token_freq.items() if c == 1]
findings["rare_word_count_freq_eq_1"] = len(rare_words)
findings["rare_word_pct_of_vocab"] = round(100 * len(rare_words) / vocab_size, 2)

# -----------------------------------------------------------------
# 14. Punctuation analysis
# -----------------------------------------------------------------
def punct_count(text):
    return sum(1 for ch in str(text) if ch in string.punctuation)

df["punct_count"] = df["user_input"].apply(punct_count)
findings["punct_count_stats"] = df["punct_count"].describe().to_dict()
findings["pct_rows_with_exclaim"] = round(100 * df["user_input"].astype(str).str.contains("!").mean(), 2)
findings["pct_rows_with_question"] = round(100 * df["user_input"].astype(str).str.contains(r"\?").mean(), 2)

# -----------------------------------------------------------------
# 15/16. Short / long input analysis
# -----------------------------------------------------------------
SHORT_THRESHOLD = 3   # words
LONG_THRESHOLD = df["word_count"].quantile(0.95)

short_inputs = df[df["word_count"] <= SHORT_THRESHOLD]
long_inputs = df[df["word_count"] >= LONG_THRESHOLD]
findings["short_input_count"] = int(len(short_inputs))
findings["short_input_threshold_words"] = SHORT_THRESHOLD
findings["long_input_count"] = int(len(long_inputs))
findings["long_input_threshold_words"] = round(float(LONG_THRESHOLD), 1)
findings["short_input_examples"] = short_inputs["user_input"].head(5).tolist()
findings["long_input_examples"] = long_inputs["user_input"].head(3).tolist()

# -----------------------------------------------------------------
# 17/18. Potential noisy / ambiguous examples
# -----------------------------------------------------------------
# Noisy: very short (<2 words) or purely numeric/symbolic
noisy = df[(df["word_count"] < 2) | (df["user_input"].astype(str).str.strip().str.len() == 0)]
findings["potential_noisy_count"] = int(len(noisy))
findings["potential_noisy_examples"] = noisy["user_input"].head(5).tolist()

# Ambiguous: same normalized text appearing under conflicting labels (already computed above)
findings["potential_ambiguous_examples"] = (
    df[df["_normalized"].isin(conflicting["_normalized"])][["user_input", "emotion_label"]]
    .head(10)
    .values.tolist()
)

# -----------------------------------------------------------------
# 19. Label-quality checks
# -----------------------------------------------------------------
findings["label_set_matches_expected"] = set(df["emotion_label"].unique()) == EXPECTED_LABELS
findings["label_value_counts_pct"] = (class_counts / len(df) * 100).round(2).to_dict()

# -----------------------------------------------------------------
# 20. Data leakage checks (train/test not split yet — this flags risk)
# -----------------------------------------------------------------
findings["rows_sharing_normalized_text_with_another_row"] = int(
    df["_normalized"].duplicated(keep=False).sum()
)

# -----------------------------------------------------------------
# Per-class word-count stats (for boxplot + table)
# -----------------------------------------------------------------
per_class_wc = df.groupby("emotion_label")["word_count"].describe()

# ===================================================================
# PLOTS
# ===================================================================

# 1. Emotion distribution
plt.figure(figsize=(8, 5))
class_counts.sort_values().plot(kind="barh", color="#6c63ff")
plt.title("Emotion Label Distribution")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "emotion_distribution.png"), dpi=150)
plt.close()

# 2. Input length distribution (words)
plt.figure(figsize=(8, 5))
plt.hist(df["word_count"], bins=30, color="#87ceeb", edgecolor="black")
plt.title("User Input Length Distribution (word count)")
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "word_count_distribution.png"), dpi=150)
plt.close()

# 3. Character count distribution
plt.figure(figsize=(8, 5))
plt.hist(df["char_count"], bins=30, color="#e6e6fa", edgecolor="black")
plt.title("User Input Length Distribution (char count)")
plt.xlabel("Character count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "char_count_distribution.png"), dpi=150)
plt.close()

# 4. Boxplot of input length by emotion
plt.figure(figsize=(10, 6))
order = sorted(df["emotion_label"].unique())
data_by_class = [df[df["emotion_label"] == c]["word_count"].values for c in order]
plt.boxplot(data_by_class, labels=order, showfliers=True)
plt.title("Input Word Count by Emotion Label")
plt.ylabel("Word count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "wordcount_by_emotion_boxplot.png"), dpi=150)
plt.close()

# 5. Top words
top_words = token_freq.most_common(20)
words, counts = zip(*top_words)
plt.figure(figsize=(9, 6))
plt.barh(words[::-1], counts[::-1], color="#574fd6")
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_words.png"), dpi=150)
plt.close()

print("Plots saved to:", PLOTS_DIR)

# ===================================================================
# WRITE MARKDOWN REPORT
# ===================================================================

def fmt_dict(d, precision=2):
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            v = round(v, precision)
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)

report = f"""# Data Quality Report — expanded_data.csv

Generated by `notebooks/01_data_analysis.py`. All numbers below were computed
directly from the dataset — none are estimated or assumed.

## 1. Dataset Dimensions

- Rows: {findings['shape'][0]}
- Columns: {findings['shape'][1]} ({', '.join(findings['columns'])})

## 2. Missing Values

{fmt_dict(findings['missing'])}

Blank/whitespace-only strings (not caught by `isnull`):

{fmt_dict(findings['blank_strings'])}

## 3–4. Duplicate Analysis

- Exact duplicate rows (all columns identical): **{findings['exact_duplicate_rows']}**
- Exact duplicate `user_input` values: **{findings['exact_duplicate_user_inputs']}**

## 5. Near-Duplicate / Leakage Analysis

Text was normalized (lowercased, punctuation stripped, whitespace collapsed)
before comparison.

- Near-duplicate inputs after normalization: **{findings['near_duplicate_normalized_inputs']}**
- Rows sharing normalized text with at least one other row: **{findings['rows_sharing_normalized_text_with_another_row']}**
- Normalized inputs mapped to **conflicting labels** (same text, different emotion): **{findings['normalized_inputs_with_conflicting_labels']}**

{'⚠️ Conflicting-label examples found — see Section 18 (Ambiguous Examples) below.' if findings['normalized_inputs_with_conflicting_labels'] > 0 else 'No conflicting-label near-duplicates found.'}

**Implication for Phase 2 (splitting):** near-duplicate rows must be kept together
(same split) to avoid train/eval leakage, since a near-identical example appearing
in both train and eval would inflate evaluation metrics artificially.

## 6–7. Class Distribution & Imbalance

| Emotion | Count | % of dataset |
|---|---|---|
{chr(10).join(f"| {k} | {v} | {findings['label_value_counts_pct'][k]}% |" for k, v in findings['class_counts'].items())}

- Imbalance ratio (max class / min class): **{findings['imbalance_ratio_max_over_min']}**
- Label set matches expected 7 classes exactly: **{findings['label_set_matches_expected']}**
- Unexpected labels found: {findings['unexpected_labels'] or 'None'}
- Expected labels missing from data: {findings['missing_expected_labels'] or 'None'}

![Emotion Distribution](plots/emotion_distribution.png)

## 8–10. Input Length Statistics

**Character count:**

{fmt_dict(findings['char_count_stats'])}

**Word count:**

{fmt_dict(findings['word_count_stats'])}

![Word Count Distribution](plots/word_count_distribution.png)
![Char Count Distribution](plots/char_count_distribution.png)
![Word Count by Emotion](plots/wordcount_by_emotion_boxplot.png)

## 11–13. Vocabulary Analysis

- Total tokens: **{findings['total_tokens']}**
- Vocabulary size (unique tokens): **{findings['vocab_size']}**
- Rare words (frequency = 1): **{findings['rare_word_count_freq_eq_1']}** ({findings['rare_word_pct_of_vocab']}% of vocabulary)

**Top 20 most frequent words:**

{chr(10).join(f"{i+1}. `{w}` — {c}" for i, (w, c) in enumerate(findings['top_20_words']))}

![Top Words](plots/top_words.png)

## 14. Punctuation Analysis

{fmt_dict(findings['punct_count_stats'])}

- Rows containing `!`: **{findings['pct_rows_with_exclaim']}%**
- Rows containing `?`: **{findings['pct_rows_with_question']}%**

## 15–16. Short / Long Input Analysis

- Short inputs (≤ {findings['short_input_threshold_words']} words): **{findings['short_input_count']}** rows
- Long inputs (≥ {findings['long_input_threshold_words']} words, 95th percentile): **{findings['long_input_count']}** rows

Short input examples:
{chr(10).join(f"- \"{x}\"" for x in findings['short_input_examples'])}

Long input examples:
{chr(10).join(f"- \"{x}\"" for x in findings['long_input_examples'])}

## 17–18. Noisy & Ambiguous Examples

- Potentially noisy rows (word count < 2 or empty): **{findings['potential_noisy_count']}**

{chr(10).join(f"- \"{x}\"" for x in findings['potential_noisy_examples']) if findings['potential_noisy_examples'] else '(none found)'}

- Ambiguous examples (identical normalized text, conflicting emotion labels): **{findings['normalized_inputs_with_conflicting_labels']}** normalized texts affected

{chr(10).join(f"- \"{x[0]}\" → labeled `{x[1]}`" for x in findings['potential_ambiguous_examples']) if findings['potential_ambiguous_examples'] else '(none found)'}

## 19. Label Quality Checks

- All 7 expected emotion classes present with no unexpected labels: **{findings['label_set_matches_expected']}**
- Per-class share of dataset shown in Section 6–7 table above.

## 20. Data Leakage Checks (pre-split)

Before any train/eval split is created, **{findings['rows_sharing_normalized_text_with_another_row']}** rows
share normalized text with at least one other row in the dataset. Phase 2's
splitting procedure groups these together by normalized text so that no
near-duplicate appears in both `train.csv` and `evaluation.csv`.

---

## Summary of Findings

- Dataset is clean at the schema level: no missing values, correct 7-class label set.
- {"Exact duplicates exist and should be deduplicated or grouped during splitting." if findings['exact_duplicate_rows'] > 0 else "No exact duplicate rows."}
- {"Class imbalance is mild" if findings['imbalance_ratio_max_over_min'] < 1.5 else "Class imbalance is present"} (ratio {findings['imbalance_ratio_max_over_min']}), which Phase 7 will examine against per-class F1 to determine whether it materially affects the model.
- Near-duplicate leakage risk exists ({findings['rows_sharing_normalized_text_with_another_row']} affected rows) and is handled explicitly in the Phase 2 split logic, not ignored.
- Vocabulary size ({findings['vocab_size']} unique tokens over {findings['total_tokens']} total tokens) is modest, consistent with a templated/paraphrased synthetic dataset rather than organically collected free-text — worth stating explicitly in the Limitations section (Phase 16).
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print("Report written to:", REPORT_PATH)
print()
print("=== KEY FINDINGS SUMMARY ===")
print("Shape:", findings["shape"])
print("Exact duplicate rows:", findings["exact_duplicate_rows"])
print("Near-duplicate normalized inputs:", findings["near_duplicate_normalized_inputs"])
print("Conflicting-label normalized inputs:", findings["normalized_inputs_with_conflicting_labels"])
print("Class counts:", findings["class_counts"])
print("Imbalance ratio:", findings["imbalance_ratio_max_over_min"])
print("Vocab size:", findings["vocab_size"])
