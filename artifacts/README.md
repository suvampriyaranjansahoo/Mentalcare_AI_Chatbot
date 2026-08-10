# Artifacts

This directory mirrors key outputs from `evaluation/` and `reports/` for
convenient artifact versioning:

- `models/` — fitted model files (`final_model.joblib`: TF-IDF vectorizer +
  Linear SVM classifier, the actual model the Flask app loads at runtime)
- `metrics/` — copies of all JSON result files from `evaluation/`
- `plots/` — copies of all generated PNG plots from `reports/plots/`
- `reports/` — copies of all markdown reports from `reports/`

Large model binaries (e.g. a fine-tuned transformer checkpoint, if you run
`notebooks/04_transformer_training.py`) are intentionally excluded via
`.gitignore` — do not commit multi-hundred-MB files to Git without LFS.
To reproduce `models/final_model.joblib` from scratch, run the pipeline in
the order documented in the root `README.md`.
