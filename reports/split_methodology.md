# Data Split Methodology

## Procedure

1. Each `user_input` was normalized (lowercased, punctuation stripped, whitespace
   collapsed) to identify near-duplicate groups, per the Phase 1 audit.
2. Rows were grouped by normalized text — 3851 unique normalized groups
   found across 4000 total rows.
3. Groups (not individual rows) were split 80/20 using
   `sklearn.model_selection.train_test_split` with `stratify=emotion_label`
   and `random_state=42`.
4. All original rows belonging to a group were assigned to whichever split
   that group was assigned to — guaranteeing no near-duplicate appears in
   both `train.csv` and `evaluation.csv`.
5. Overlap between the two splits was verified programmatically (assertion
   in `02_data_split.py`) — **0 overlapping normalized texts** confirmed.

## Configuration

- Random seed: `42`
- Target evaluation fraction: `0.2` (of unique normalized groups)
- Stratification column: `emotion_label`

## Resulting Split Sizes

- `data/train.csv`: **3200** rows
- `data/evaluation.csv`: **800** rows
- Actual eval fraction (rows): **0.2**

## Class Distribution After Split

| Emotion | Train count | Train % | Eval count | Eval % |
|---|---|---|---|---|
| anger | 459 | 14.34% | 112 | 14.0% |
| disgust | 455 | 14.22% | 116 | 14.5% |
| fear | 455 | 14.22% | 116 | 14.5% |
| joy | 457 | 14.28% | 115 | 14.38% |
| neutral | 457 | 14.28% | 115 | 14.38% |
| sadness | 459 | 14.34% | 113 | 14.12% |
| surprise | 458 | 14.31% | 113 | 14.12% |

## Guarantee

`data/evaluation.csv` is held out from this point forward. It must not be used
for baseline model selection (Phase 3), transformer hyperparameter tuning, or
FAQ threshold selection (Phase 8) — only for final reported metrics (Phase 5,
Phase 17). Model/threshold selection uses cross-validation on `train.csv` only.
