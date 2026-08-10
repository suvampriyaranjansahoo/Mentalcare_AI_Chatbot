# Model Comparison — Statistical Rigor & Reproducibility

## Confidence Intervals on Cross-Validation Results

The 5-fold cross-validation in Phase 3 (`evaluation/baseline_results.json`)
reports mean ± standard deviation across folds for every metric, rather than
a single point estimate. This is the standard, honest way to represent
variance for a small number of CV folds (n=5 is too few folds for a
normal-approximation 95% CI to be meaningful — reporting mean ± std across
the actual folds is more transparent than manufacturing a CI from 5 samples).

Example (Linear SVM, weighted F1): **1.0000 ± 0.0000** — zero variance across
all 5 folds, i.e. every fold achieved perfect classification. This is
consistent with, and further evidence for, the Phase 1 finding that the
dataset's small vocabulary (210 tokens) makes the classes linearly separable.

## Statistical Comparison Between Models

Because two of the three baselines (Logistic Regression: 0.9997, Linear SVM:
1.0000) are separated by a difference that is smaller than typical
fold-to-fold noise on a harder dataset, and because the third (Naive Bayes:
0.9991) differs from the top model by only 0.0009, a formal significance
test (e.g. paired t-test across folds, or McNemar's test on held-out
predictions) was **not performed**, since the practical difference between
99.9% and 100% is not meaningful for model selection — all three baselines
solve this particular dataset essentially perfectly. A significance test
would be appropriate and necessary if this pipeline is re-run against a
harder, more naturalistic dataset (see recommendation in `error_analysis.md`
and `LIMITATIONS.md`), where real separation between models is expected.

## Reproducibility

| Component | Value |
|---|---|
| Random seed (all scripts) | `42` |
| Data split | 80% train / 20% evaluation, stratified by `emotion_label`, grouped by normalized text to prevent near-duplicate leakage (Phase 2) |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Preprocessing | `TfidfVectorizer(ngram_range=(1,2), min_df=2)` — no external tokenizer, no stopword removal applied (dataset already short/informal) |
| Baseline hyperparameters | Logistic Regression: `max_iter=1000`, default C; Linear SVM: default C; Naive Bayes: default `alpha` — no hyperparameter search was performed since default settings already reach ~100% (see Phase 3 note); a search would be warranted on a harder dataset |
| Model/tokenizer version | scikit-learn baselines use whatever `scikit-learn` version is pinned in `requirements.txt`. Transformer (Phase 4, pending local run): `j-hartmann/emotion-english-distilroberta-base`, exact revision should be pinned by whoever runs `notebooks/04_transformer_training.py` and recorded in `evaluation/transformer_results.json` |
| Evaluation protocol | Model/threshold selection uses `train.csv` only (CV or internal train/val split); `evaluation.csv` is touched exactly once per model, for the final reported metric only |

## How Another Developer Reproduces These Results

```bash
pip install -r requirements.txt
python notebooks/01_data_analysis.py       # data audit + report
python notebooks/02_data_split.py          # train/eval split (deterministic, seed=42)
python notebooks/03_baseline_models.py     # baseline CV results
python notebooks/05_model_comparison.py    # final model selection + held-out eval
python notebooks/06_error_analysis.py      # confusion matrix + error report
python notebooks/07_class_imbalance.py     # imbalance analysis
python notebooks/08_faq_experiment.py      # FAQ threshold experiment
python notebooks/09_conversation_analytics.py  # demo analytics
```

All scripts are deterministic given the same input CSV and the fixed
`random_state=42` used throughout. Running them in this order on the same
`data/expanded_data.csv` will reproduce every number in this repository's
reports exactly.

The transformer step (`notebooks/04_transformer_training.py`) requires
network access to `huggingface.co` to download model weights, which was
unavailable in the environment that produced this repository. It must be run
separately; see that script's docstring for details.
