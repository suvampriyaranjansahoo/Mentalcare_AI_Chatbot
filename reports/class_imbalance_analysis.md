# Class Imbalance Analysis

## Dataset Balance (from Phase 1 audit)

The Phase 1 data quality report measured a class imbalance ratio of **1.002**
(max class / min class), i.e. the dataset is essentially perfectly balanced
across all 7 emotion classes (~571-572 rows each before splitting). This is
consistent with the dataset being synthetically constructed rather than
collected from an organic, naturally imbalanced source.

## Per-Class Performance (held-out evaluation set)

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 1.0000 | 1.0000 | 1.0000 | 112 |
| disgust | 1.0000 | 1.0000 | 1.0000 | 116 |
| fear | 1.0000 | 1.0000 | 1.0000 | 116 |
| joy | 1.0000 | 1.0000 | 1.0000 | 115 |
| neutral | 1.0000 | 1.0000 | 1.0000 | 115 |
| sadness | 1.0000 | 1.0000 | 1.0000 | 113 |
| surprise | 1.0000 | 1.0000 | 1.0000 | 113 |

- **Macro F1:** 1.0
- **Weighted F1:** 1.0

Since macro F1 (unweighted across classes) and weighted F1 (weighted by
class support) are equal here (1.0 vs 1.0),
there is no meaningful gap between them — the expected signature of a
well-balanced dataset with no per-class degradation.

## Conclusion

Class imbalance is **not a meaningful factor** in this project, because the
dataset was constructed to be balanced from the start (Phase 1 finding).
Class weighting or resampling strategies were considered but **not applied**,
since there is no imbalance to correct and applying them would only add
unnecessary complexity without a validated benefit — consistent with the
instruction not to add such techniques "simply for appearance."

If this pipeline is later applied to a naturally-collected, imbalanced
dataset (see Phase 6's recommendation to test against a public benchmark),
this analysis should be re-run, since imbalance effects may become visible
there.
