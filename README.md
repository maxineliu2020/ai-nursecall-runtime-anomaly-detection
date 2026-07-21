# Runtime Anomaly Detection for Nurse-Call-Like Service Logs

Reproducibility package for the revised Scientific Reports manuscript:

> **Simulation-Based Runtime Anomaly Detection for Nurse Call System Assurance: A Reproducible Proof of Concept Using Public Service-Request Logs**

This repository contains two explicitly separated computational pipelines:

- **Pipeline B (primary):** transforms a fixed NYC 311 service-request snapshot into a non-clinical, nurse-call-like service-log proxy and evaluates delay-focused anomaly detection.
- **Pipeline A (supplementary):** generates fully synthetic event logs and evaluates detectability across controlled anomaly types.

No real nurse-call, patient, staffing, or clinical workflow data are used. The software is a research proof of concept, not a medical device or deployment-ready clinical monitor.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21471817.svg)](https://doi.org/10.5281/zenodo.21471817)

## Canonical release

- Release: `v2.0.2`
- Version DOI: assigned automatically when Zenodo archives the `v2.0.2` GitHub release
- Concept DOI: `10.5281/zenodo.17767142` (resolves to the latest release)

Before publishing a new release, verify that the release archive contains the complete `outputs/`, `figures_v2/`, code, environment file, and data snapshot described below. A new Zenodo version receives a new version DOI; the concept DOI remains unchanged.

## Repository layout

```text
.
├── DATA/
│   ├── erm2-nwe9.csv
│   ├── 311_ServiceRequest_2010-Present_DataDictionary_Updated_2023.xlsx
│   └── README.md
├── src/
│   ├── experiment_v2.py
│   ├── experiment_synth_v2.py
│   └── make_figures_v2.py
├── streamlit/
│   ├── app.py
│   └── small_demo.csv
├── outputs/
│   ├── pipeline_b/
│   └── pipeline_a/
├── figures_v2/
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md
```

Files from the superseded v1 protocol, including `experiment.py`, `experiment_real_plus.py`, and their figures, should be moved to `legacy/v1/` or omitted from the v2 release archive. They must not be presented as canonical v2 results because they use different splits, thresholding, cleaning, and evaluation procedures.

## Data integrity

The canonical snapshot contains 100,000 records. Its expected SHA-256 digest is:

```text
bf29e88716efe5cb4d6fd26e3fc4d46edf05f6545ea3481e99c46199197b97c0
```

Verify it before running:

```bash
python -c "import hashlib, pathlib; p=pathlib.Path('DATA/erm2-nwe9.csv'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Pipeline B derives response time, hour, weekday, and category indicators at run time. The required source columns are `created_date`, `closed_date`, and `agency`.

## Environment

Use Python 3.12.3 and the pinned `requirements.txt` from the release. The manuscript’s canonical environment reports:

- NumPy 2.4.4
- pandas 3.0.2
- scikit-learn 1.8.0
- SciPy 1.17.1
- SHAP 0.52.0

Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the closest match to the manuscript environment, Docker is recommended. The
compose service fixes Python at 3.12.3 and limits the analysis to one CPU:

```bash
docker compose run --rm reproduce
```

The repository is mounted into the container, so regenerated outputs and figures
are written back to `outputs/` and `figures_v2/` on the host.

## Reproduce the manuscript analyses

Run from the repository root. All default paths are repository-relative; no user-specific or container-specific directories are required.

### Pipeline B — primary analysis

```bash
python src/experiment_v2.py
```

Equivalent explicit invocation:

```bash
python src/experiment_v2.py \
  --data DATA/erm2-nwe9.csv \
  --output-dir outputs/pipeline_b \
  --seeds 101,102,103,104,105,106,107,108,109,110
```

This command produces:

- primary full-feature and leakage-ablation summaries and per-seed results;
- the paired target-leakage F1 and average-precision comparisons reported in the manuscript;
- timeout and prevalence sensitivity outputs;
- fixed-model base-rate control outputs;
- 1% prevalence retraining without `resp_h`;
- positive-subsample composition checks;
- `supplementary_checks.json` linking the new audit claims to their source files;
- data, split, environment, and runtime metadata.

### Pipeline A — supplementary synthetic analysis

```bash
python src/experiment_synth_v2.py
```

The script overwrites canonical summaries deterministically; repeated execution does not append duplicate rows.

### Figures

After both pipelines complete:

```bash
python src/make_figures_v2.py
```

The script generates exactly 17 publication figures (9 main and 8 supplementary), `shap_metadata.json`, and `figure_manifest.json` with SHA-256 checksums. Figures are saved at 300 dpi by default.

## Key audit outputs

The final manuscript’s counterintuitive low-prevalence findings are supported by:

```text
outputs/pipeline_b/supplementary_checks.json
outputs/pipeline_b/fixed_model_control.csv
outputs/pipeline_b/fixed_model_control_runs.csv
outputs/pipeline_b/prevalence_1pct_no_resp.csv
outputs/pipeline_b/prevalence_1pct_no_resp_runs.csv
outputs/pipeline_b/prevalence_sensitivity.csv
outputs/pipeline_b/prevalence_sensitivity_runs.csv
```

Do not report these claims from hand-entered values. Regenerate them from the archived snapshot and retain the per-seed files in the DOI release.

## Interactive demonstration

The Streamlit application is a simplified educational companion:

```bash
streamlit run streamlit/app.py
```

The demo is not the manuscript experiment. It intentionally does not reproduce the full multi-seed protocol, and its displayed results must not be substituted for canonical outputs.

## Release checklist

Before creating a GitHub release and Zenodo version:

1. Run both pipelines and the figure script from a clean environment.
2. Confirm `supplementary_checks.json` exists and contains the expected data SHA-256.
3. Confirm `figure_manifest.json` lists exactly 17 figures.
4. Check that no canonical source file contains `/home/`, `/mnt/`, a username, or another machine-specific absolute path.
5. Move obsolete scripts and old figures to `legacy/v1/` or exclude them from the release.
6. Record the final Git commit in the manuscript, response letter, and Zenodo metadata.
7. Cite the new version DOI, not an earlier immutable Zenodo version.

Then run the automated release gate:

```bash
python src/validate_release.py
```

## License and attribution

See `LICENSE` for software licensing and `DATA/sources.md` for NYC Open Data provenance and terms. Cite both the manuscript and the version-specific Zenodo record when reusing this package.

## Citation

Machine-readable software citation metadata are provided in `CITATION.cff` and
Zenodo release metadata in `.zenodo.json`. For reproducibility, cite the
version-specific DOI shown on the Zenodo record for `v2.0.2`. The concept DOI
<https://doi.org/10.5281/zenodo.17767142> always resolves to the latest release.
