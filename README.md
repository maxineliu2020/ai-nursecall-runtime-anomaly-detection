# Runtime Anomaly Detection for Nurse-Call-like Service Logs

Reproducibility repository for:

> **Simulation-Based Runtime Anomaly Detection for Nurse Call System Assurance: A Reproducible Proof of Concept Using Public Service-Request Logs**
> Yuanyuan Liu and David R. Concepcion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21466809.svg)](https://doi.org/10.5281/zenodo.21466809)

## What this repository contains

Two experimental pipelines, both fully reproducible from a single command:

| Pipeline | Script | Data | Purpose |
|---|---|---|---|
| **B — 311 transformation (primary)** | `src/experiment_v2.py` | `DATA/erm2-nwe9.csv` | Main evidence: delay-defined anomaly detection on transformed NYC 311 service-request logs |
| **A — synthetic injection (supplementary)** | `src/experiment_synth_v2.py` | generated per seed | Controlled five-type anomaly injection with per-type recall analysis |

The **v2 protocol** (this release) uses a 60/20/20 train/validation/test split,
validation-only threshold selection, 10 independent seeds with t-interval 95% CIs,
a target-leakage feature ablation, a delay-label timeout sweep, fixed-prevalence
subsampling, and runtime benchmarks. See `CHANGELOG.md` for differences from v1.0.1.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/experiment_v2.py          # pipeline B: main table, sweep, sensitivity, benchmark
python src/experiment_synth_v2.py    # pipeline A: synthetic injection suite
python src/make_figures_v2.py        # regenerate all manuscript figures (300 dpi)
```

Outputs are written to `results_v2/`, `results_synth_v2/`, and `figures_v2/`.
Pre-computed outputs from the canonical run are committed for reference.

## Data provenance

`DATA/erm2-nwe9.csv` is the exact snapshot used in the manuscript: the most recent
100,000 records of the NYC Open Data *311 Service Requests* dataset
(dataset id `erm2-nwe9`) at download time; `created_date` spans
2025-10-16 17:35:07 to 2025-10-26 01:49:41. Cleaning (documented in the paper and
implemented in `experiment_v2.py`): 24,671 records with missing closure timestamps
and 8 records with negative response intervals are excluded, retaining 75,321
records. Column semantics: `DATA/311_ServiceRequest_...DataDictionary...xlsx`.

This is **non-clinical public service-request data** used as a structural proxy for
timestamped service logs. It contains no patient, staffing, or clinical information.

## Environment

Canonical results were produced with Python 3.12.3 on Linux x86_64 (1 vCPU):
`numpy==2.4.4  pandas==3.0.2  scikit-learn==1.8.0  scipy==1.17.1  shap==0.52.0`.
All seeds are fixed (pipeline B: 101–110; pipeline A: 201–210).

## Interactive demo

`streamlit run streamlit/app.py` — simplified companion demo. It applies the same
cleaning rules as the paper, restricts features to a whitelist
(`resp_h`, `hour`, `weekday`, `is_weekend`), and treats only a column literally
named `is_anomaly` as ground truth.

## Repository structure

```
DATA/                 input snapshot + data dictionary + provenance notes
src/                  experiment_v2.py, experiment_synth_v2.py, make_figures_v2.py
results_v2/           canonical pipeline-B outputs (CSV/JSON)
results_synth_v2/     canonical pipeline-A outputs
figures_v2/           manuscript figures (300 dpi PNG)
streamlit/            demo app
archive_v1/           v1.0.1 scripts and outputs, retained read-only for provenance
```

## Authors

- **Yuanyuan (Maxine) Liu** — Johns Hopkins University; Nova Southeastern University
  (corresponding: yliu536@jh.edu, ORCID 0000-0003-3410-6893)
- **David R. Concepcion** — Johns Hopkins University

## Citation

If you use this code or data snapshot, please cite the paper (reference to be
updated upon publication) and the archived software release
(doi:10.5281/zenodo.17767142, all versions).

## License

MIT — see `LICENSE`.
