# Changelog

## v2.0.0 — Scientific Reports revision (2026-07)

Methodological revision responding to peer review. Not backward compatible with
v1.0.1 result interpretation.

### Protocol changes (pipeline B, `experiment_v2.py`, replaces `experiment_real_plus.py`)
- 60/20/20 train/validation/test split (stratified, per-seed); v1 used 70/30 with
  threshold selection on the test set.
- Thresholds selected on the validation set only, by maximizing F1 (documented as
  label-informed retrospective selection); label-free alert-budget thresholds
  (top 10% / 20% of training scores) reported alongside.
- Fixed the threshold–recall pairing defect in v1's `threshold_at_recall`.
- Explicit data cleaning: 24,671 missing-closure records treated as right-censored
  and excluded from delay-label evaluation (v1 silently labeled them normal);
  8 negative-interval records excluded as invalid timestamps; 2,319 zero-duration
  administrative closures retained and documented (robustness check with them
  excluded: IF F1 0.769 / AP 0.815 vs 0.751 / 0.760).
- 10 independent seeds; mean, SD, and t-interval 95% CIs over runs (v1 reported a
  degenerate bootstrap over a single metric value).
- New analyses: feature ablation with/without `resp_h` (target-leakage exposure,
  paired t-test across seeds: ΔF1 = 0.064, p = 3.3e-11); delay-label timeout sweep
  (1–48 h); fixed-prevalence subsampling (1/2/5/10%); per-category-quantile rule
  baseline; runtime benchmarks (latency, throughput, memory, model size).
- One-Class SVM trained on a 15,000-record seeded subsample per run (tractability;
  evaluated in the main configuration only).
- Autoencoder baseline removed.

### Pipeline A (`experiment_synth_v2.py`, replaces `experiment.py`)
- Injection design unchanged from v1 (same distributions and parameters).
- Evaluation corrected: per-seed independent data generation, 60/20/20 split,
  validation-only thresholds, full metrics, per-anomaly-type recall, inference-only
  latency (v1 timed training + inference, used median-score thresholds, and
  evaluated on training data).
- Effective prevalence now reported (injection rates 5/10/20% correspond to
  15.6/27.0/44.1% prevalence due to injected extra rows).

### Data & repository
- `DATA/erm2-nwe9.csv` snapshot committed (63 MB) with documented provenance.
- v1 scripts and outputs moved to `archive_v1/` (read-only provenance).
- `requirements.txt` pinned to the canonical environment.
- Streamlit demo v2: strict `is_anomaly` ground-truth detection, feature whitelist,
  paper-consistent cleaning, timeout-based labeling.
- Removed committed `venv/` directory and stray binaries.

## v1.0.1 — Initial release (2025-11-30)
- Initial public release archived at doi:10.5281/zenodo.17767143.
