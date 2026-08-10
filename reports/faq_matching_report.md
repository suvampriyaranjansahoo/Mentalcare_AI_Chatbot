# FAQ Fuzzy Matching Experiment

FAQ bank: `train.csv` (3200 entries). Validation queries: `evaluation.csv` (800 entries).

## Similarity Score Distributions

| Approach | Mean best-match score | Std | Min | Max |
|---|---|---|---|---|
| difflib_character_similarity | 0.899 | 0.0321 | 0.7463 | 0.9492 |
| token_jaccard_similarity | 0.783 | 0.1076 | 0.25 | 0.9167 |
| tfidf_cosine_similarity | 0.8856 | 0.0599 | 0.4549 | 0.9827 |

## Evidence-Based Threshold Selection

Selection rule: among thresholds where at least 50% of queries still get matched, pick the threshold with the highest match accuracy (not an arbitrary fixed value like 0.80).

| Approach | Chosen threshold | Match accuracy | False-match rate | Unmatched rate |
|---|---|---|---|---|
| difflib_character_similarity | 0.9 | 0.994 | 0.0037 | 0.38 |
| token_jaccard_similarity | 0.8 | 1.0 | 0.0 | 0.4375 |
| tfidf_cosine_similarity | 0.6 | 1.0 | 0.0 | 0.0088 |

![Threshold Sweep](plots/faq_threshold_sweep.png)

## Interpretation

Because this dataset is templated with limited vocabulary (see Phase 1 finding: 210 unique tokens), all three similarity approaches likely perform very well at matching queries back to their originating template family, in the same way the classifier baselines did. This is an honest measurement, but the same caveat applies as in the model comparison: these numbers reflect performance on templated text, not necessarily on naturalistic free-form user input the way the original project's live chatbot would receive it in production.

The original project used `difflib.get_close_matches` with a fixed `cutoff=0.5`. Comparing that fixed value against the evidence-based thresholds chosen above shows whether 0.5 was a reasonable choice or arbitrary luck.