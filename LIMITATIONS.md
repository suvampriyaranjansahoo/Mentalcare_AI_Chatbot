# Limitations

This project is a **portfolio/demo application**, not a clinical or
production mental-health tool. The following limitations are documented
explicitly and should be read alongside any metric reported elsewhere in
this repository.

## Dataset Limitations

- **Small, synthetic dataset.** 4,000 rows, generated via LLM prompting
  rather than collected from real users or a peer-reviewed corpus.
- **English-only.** No multilingual support or evaluation.
- **Very limited vocabulary.** The Phase 1 audit measured only 210 unique
  tokens across the entire dataset, and 149 near-duplicate inputs after
  text normalization. This means the dataset is built from a small number
  of sentence templates with word substitutions, not organically diverse
  free-form text.
- **Consequence for all reported metrics:** because of the above, the 7
  emotion classes are close to linearly separable by surface vocabulary
  alone. Every baseline model in this repository reaches 99.9-100% accuracy
  on both cross-validation and the held-out evaluation set. This is a real,
  measured result, but it demonstrates that **the pipeline and methodology
  are implemented and evaluated correctly** — it does **not** demonstrate
  that the underlying model architecture would perform this well on
  naturalistic, diverse, real-world emotional text. See
  `reports/error_analysis.md` for the full discussion.
- **Seven-class emotion taxonomy is simplified.** Real emotional expression
  is more nuanced, layered, and often mixed than any single-label 7-way
  classification can represent.
- **Conversational text can be genuinely ambiguous.** Sarcasm, indirect
  expression, and mixed emotion are not well represented in this synthetic
  dataset, and the current model has not been evaluated against text with
  those properties.

## Model Limitations

- Model predictions **can be incorrect**, especially on text meaningfully
  different in style or vocabulary from the training distribution (a
  strong possibility given the dataset's narrow vocabulary).
- Fuzzy/FAQ matching (Phase 8) **can produce false matches** — the
  evidence-based threshold reduces but does not eliminate this risk, and
  the false-match rate reported in `reports/faq_matching_report.md` should
  be reviewed rather than assumed to be zero in production.
- Generated responses (from the emotion-based template bank) require the
  quality-gate safeguard implemented in `flask_app/pipeline.py` — an
  unfiltered generative model was **not** used precisely because
  unconstrained generation in a mental-health-adjacent context carries
  real risk of producing inappropriate or unsafe output.
- The transformer model (`j-hartmann/emotion-english-distilroberta-base`,
  Phase 4) has **not yet been fine-tuned or evaluated** in this repository
  due to a sandboxed environment without `huggingface.co` access. The
  script to do so is provided (`notebooks/04_transformer_training.py`) but
  its results are not yet part of this repository's reported metrics.

## Application Limitations

- **This is not a clinical tool.** It performs no diagnosis, no treatment
  recommendation, and has no crisis-intervention functionality.
- It should never be relied upon as a substitute for professional mental
  health support, and the application UI includes a visible disclaimer to
  this effect.
- Conversation analytics (Phase 10) are demonstrated on **simulated demo
  traffic**, not real user data, since the application has not been
  deployed with real users. The analytics code is real and will work
  identically on real logs once deployed — only the current input data is
  synthetic.
- No authentication, rate limiting, or abuse protection has been
  implemented — this is out of scope for a portfolio demo but would be
  required before any real deployment handling real user data.

## What Would Be Needed to Make Metrics More Meaningful

1. Evaluate the model (baseline and/or fine-tuned transformer) against a
   naturalistic, peer-reviewed emotion-text benchmark (e.g., a public
   dataset such as `dair-ai/emotion` or GoEmotions) in addition to this
   synthetic dataset, and report both results side by side.
2. Collect or generate a harder, more linguistically diverse dataset with
   genuine class overlap, sarcasm, and mixed emotion, so that error
   analysis (Phase 6) can surface real, informative failure patterns
   rather than reporting a 0% error rate.
3. Run a live pilot to collect real conversational logs and re-run the
   analytics pipeline (Phase 10) on real data.
