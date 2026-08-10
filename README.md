# MENTALCARE AI — Emotion-Aware Support Chatbot

A hybrid NLP system combining TF-IDF-based FAQ retrieval with a fine-tuned
emotion classifier, built and evaluated as a rigorous ML methodology
exercise rather than a UI showcase. **Not a clinical tool** — see
[`LIMITATIONS.md`](LIMITATIONS.md).

## Problem Statement

### Business / User Problem

Support and companionship chatbots often either (a) rely purely on rigid
FAQ scripts that fail on any paraphrase, or (b) rely purely on generative
models with no guardrails, which is risky in an emotionally sensitive
context. This project builds and evaluates a **hybrid retrieval + ML
architecture** that combines the reliability of curated FAQ responses with
the flexibility of a learned emotion classifier, backed by a real,
documented evaluation methodology.

### Dataset

`data/expanded_data.csv` — 4,000 rows, 3 columns (`user_input`,
`emotion_label`, `bot_response`), covering 7 emotion classes: joy, sadness,
anger, fear, surprise, disgust, neutral. LLM-generated for this project;
see [Data Quality Analysis](reports/data_quality_report.md) for a full
audit, including a significant limitation (narrow vocabulary) that shapes
how every downstream metric should be interpreted.

## Methodology

```
DATA → EDA → DATA QUALITY → SPLITTING → BASELINES → MODEL TRAINING
  → EVALUATION → ERROR ANALYSIS → MODEL SELECTION → HYBRID SYSTEM
  → ANALYTICS → TESTING → MLOPS → DEPLOYMENT
```

| Phase | Report |
|---|---|
| Data audit | [`reports/data_quality_report.md`](reports/data_quality_report.md) |
| Train/eval split | [`reports/split_methodology.md`](reports/split_methodology.md) |
| Baselines (CV) | [`evaluation/baseline_results.json`](evaluation/baseline_results.json) |
| Transformer | [`notebooks/04_transformer_training.py`](notebooks/04_transformer_training.py) — **script provided, not yet executed** (see below) |
| Model comparison & selection | [`reports/model_comparison.md`](reports/model_comparison.md) |
| Final held-out evaluation | [`evaluation/final_results.json`](evaluation/final_results.json) |
| Error analysis | [`reports/error_analysis.md`](reports/error_analysis.md) |
| Class imbalance | [`reports/class_imbalance_analysis.md`](reports/class_imbalance_analysis.md) |
| FAQ retrieval experiment | [`reports/faq_matching_report.md`](reports/faq_matching_report.md) |
| Conversation analytics | [`reports/conversation_analytics.md`](reports/conversation_analytics.md) |
| Statistical rigor & reproducibility | [`reports/model_comparison_statistics.md`](reports/model_comparison_statistics.md) |

## Baseline Models

Three TF-IDF-based baselines were evaluated with 5-fold stratified
cross-validation on the training split (n=3,200):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9997 ± 0.0006 | 0.9997 ± 0.0006 | 0.9997 ± 0.0006 |
| **TF-IDF + Linear SVM** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** |
| TF-IDF + Multinomial Naive Bayes | 0.9991 ± 0.0019 | 0.9991 ± 0.0019 | 0.9991 ± 0.0019 |

**Linear SVM was selected** (highest weighted F1, zero variance across
folds) and evaluated once on the held-out evaluation set (n=800):
**Accuracy 1.0000, Macro F1 1.0000, Weighted F1 1.0000.**

### Why these numbers are this high — read before citing them

The data audit found only **210 unique vocabulary tokens** across the whole
dataset. This makes the 7 classes close to linearly separable by surface
vocabulary alone, which is why even simple TF-IDF baselines reach ~100%
with zero fine-tuning or hyperparameter search. Full discussion in
[`reports/error_analysis.md`](reports/error_analysis.md) and
[`LIMITATIONS.md`](LIMITATIONS.md). **These numbers demonstrate that the
methodology and pipeline are implemented and evaluated correctly — they do
not, by themselves, demonstrate strong generalization to naturalistic
real-world text.**

### Transformer model (Phase 4) — pending

`j-hartmann/emotion-english-distilroberta-base` fine-tuning requires
`huggingface.co` access unavailable in the environment that built this
repository. The complete, ready-to-run script is at
`notebooks/04_transformer_training.py`. Running it and placing the output
at `evaluation/transformer_results.json`, then re-running
`notebooks/05_model_comparison.py`, will populate the transformer row in
the comparison table with real numbers.

## Hybrid Retrieval + ML Architecture

```
User message
     │
     ▼
Input validation (non-empty, length cap)
     │
     ▼
FAQ similarity — TF-IDF cosine, threshold chosen empirically (Phase 8: 0.6,
selected via threshold sweep on held-out queries, not an arbitrary default)
     │
     ├── High-confidence match ──► Curated FAQ response
     │
     └── No match ──► Emotion classifier (TF-IDF + Linear SVM)
                            │
                       Emotion-appropriate templated response
                            │
                       Quality gate (non-empty, not an echo of input)
                            │
                       Safe fallback if gate fails
     │
     ▼
Every request logged: session_id, timestamp, input length, FAQ similarity,
predicted emotion, classifier confidence, response type, latency
```

Implementation: [`flask_app/pipeline.py`](flask_app/pipeline.py)

### FAQ Retrieval Experiment

Three similarity approaches (character-level `difflib`, token Jaccard,
TF-IDF cosine) were compared on the held-out set, sweeping thresholds
rather than assuming a fixed cutoff. **TF-IDF cosine at threshold 0.6**
was selected: 100% match accuracy with only a 0.88% unmatched rate,
substantially outperforming `difflib` (which reaches only ~62% match rate
at best on this data). Full results:
[`reports/faq_matching_report.md`](reports/faq_matching_report.md).

## Conversation Analytics

The analytics pipeline (`notebooks/09_conversation_analytics.py`) tracks
message volume, emotion distribution, FAQ-vs-fallback rate, latency, and
classifier confidence. Since this app has not been deployed with real
users, the current report runs on simulated traffic sent through the real
pipeline — the code is identical to what would run on real logs.
**These are application-level statistics, not clinical measurements.**

## Testing

19 tests across 4 files (`tests/`) covering FAQ matching, emotion
classification, Flask routes (including malformed JSON and validation
edge cases), and data pipeline integrity (split leakage checks,
stratification). Run with `pytest tests/ -v`. CI runs this automatically
on every push/PR via `.github/workflows/tests.yml`.

## MLOps

- Dependencies pinned in `requirements.txt`
- `Dockerfile` builds a container serving the Flask app on the pinned SVM
  model (no runtime dependency on `huggingface.co`)
- No hardcoded local paths — all paths resolve relative to the repo root
- `FLASK_DEBUG` and `PORT` configured via environment variables, debug
  disabled by default
- GitHub Actions runs the full data pipeline + test suite on every push

## Reproducibility

Full details: [`reports/model_comparison_statistics.md`](reports/model_comparison_statistics.md).
Fixed random seed (`42`) throughout; deterministic stratified split with
near-duplicate leakage prevention; model/threshold selection strictly
separated from the held-out evaluation set.

## Setup

```bash
git clone <this-repo>
cd MENTALCARE_AI
pip install -r requirements.txt

# Regenerate all data artifacts and reports:
python notebooks/01_data_analysis.py
python notebooks/02_data_split.py
python notebooks/03_baseline_models.py
python notebooks/05_model_comparison.py
python notebooks/06_error_analysis.py
python notebooks/07_class_imbalance.py
python notebooks/08_faq_experiment.py
python notebooks/09_conversation_analytics.py

# Run tests:
pytest tests/ -v

# Run the app:
cd flask_app && python app.py
# then open http://localhost:5000
```

Or with Docker:

```bash
docker build -t mentalcare-ai .
docker run -p 5000:5000 mentalcare-ai
```

## Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full, honest list — dataset
vocabulary size, synthetic data caveats, pending transformer evaluation,
and what this project is explicitly **not** (a clinical or diagnostic
tool).

## Disclaimer

This is a portfolio project demonstrating ML/NLP engineering methodology.
It is not a substitute for professional mental health care. If you are in
distress, please contact a licensed professional or local emergency/crisis
services.

## License

MIT — see [`LICENSE`](LICENSE).
