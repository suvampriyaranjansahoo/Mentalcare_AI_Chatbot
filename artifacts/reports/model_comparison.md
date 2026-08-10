# Model Comparison

All cross-validation numbers below come from `evaluation/baseline_results.json` (5-fold stratified CV on `data/train.csv`). The transformer row is populated only once `notebooks/04_transformer_training.py` has been run and its output copied into `evaluation/transformer_results.json`.

| Model | Accuracy | Macro F1 | Weighted F1 | Approach | Status |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9997 ± 0.0006 | 0.9997 ± 0.0006 | 0.9997 ± 0.0006 | 5-fold stratified cross-validation on train.csv (n=3200) | evaluated (5-fold CV on train.csv) |
| TF-IDF + Linear SVM | 1.0000 | 1.0000 | 1.0000 | 5-fold stratified cross-validation on train.csv (n=3200) | evaluated (5-fold CV on train.csv) |
| TF-IDF + Multinomial Naive Bayes | 0.9991 ± 0.0019 | 0.9991 ± 0.0019 | 0.9991 ± 0.0019 | 5-fold stratified cross-validation on train.csv (n=3200) | evaluated (5-fold CV on train.csv) |
| Fine-tuned j-hartmann/emotion-english-distilroberta-base | — | — | — | Not yet run in this environment (requires huggingface.co access — see notebooks/04_transformer_training.py to run locally) | PENDING — script provided, not executed |

## Selection

'TF-IDF + Linear SVM' selected: highest mean weighted F1 in 5-fold CV on train.csv (1.0000 ± 0.0000). Transformer fine-tuning (Phase 4) was not executable in this environment (no huggingface.co access); once run locally, its held-out result should be compared against this baseline before final selection is treated as permanent.

## Final Held-Out Evaluation (evaluation.csv, n=800)

- **Accuracy:** 1.0
- **Macro F1:** 1.0
- **Weighted F1:** 1.0

⚠️ **Important caveat**, carried over from the Phase 1 data audit: this dataset's vocabulary is only 210 unique tokens across 4,000 rows, generated from a small set of sentence templates. Both the CV and held-out numbers above are near-ceiling (99–100%) as a direct consequence of that — they demonstrate the pipeline and methodology work correctly, but should **not** be read as evidence of strong generalization to naturalistic, diverse real-world text. See `LIMITATIONS.md`.