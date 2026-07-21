# v2.0.0 — Revised protocol (Scientific Reports resubmission)

This release accompanies the revised manuscript and supersedes v1.0.1 for all
quantitative claims. Key changes: train/validation/test protocol with
validation-only threshold selection; 10-seed confidence intervals; explicit data
cleaning with right-censored missing-closure handling; target-leakage feature
ablation; delay-label timeout sweep; fixed-prevalence sensitivity analysis;
independent rule baseline; runtime benchmarks; corrected synthetic-injection
evaluation with per-type recall. Full details in CHANGELOG.md.

Canonical environment: Python 3.12.3, numpy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0, scipy 1.17.1, shap 0.52.0. Seeds: 101–110 (pipeline B),
201–210 (pipeline A).

Reproduce: `pip install -r requirements.txt && python src/experiment_v2.py`
