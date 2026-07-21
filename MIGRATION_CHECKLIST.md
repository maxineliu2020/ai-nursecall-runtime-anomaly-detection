# Migration checklist for the next DOI release

## Replace canonical source

Copy these files into the repository at the same relative paths:

```text
src/experiment_v2.py
src/experiment_synth_v2.py
src/make_figures_v2.py
src/validate_release.py
streamlit/app.py
streamlit/small_demo.csv
.devcontainer/devcontainer.json
.github/workflows/code-quality.yml
README.md
DATA/README.md
requirements.txt
```

## Quarantine superseded v1 material

The following files use superseded methods and must not remain beside canonical v2 outputs without an explicit legacy label:

```text
experiment.py
experiment_real_plus.py
alerts_per_hour.png
box_delay_by_category.png
category_pie.png
cm_IF.png
heatmap_weekday_hour.png
hist_delay.png
kde_delay.png
metrics_bar_ci.png
ops_alerts_vs_th_IF.png
ops_alerts_vs_th_OCSVM.png
ops_alerts_vs_th_RF.png
pr_curves_all.png
```

Either remove them from the new release or move them to `legacy/v1/` with a README stating that they are retained only for provenance. In particular:

- the old histogram and KDE include negative response intervals that the revised protocol excludes;
- the old confusion matrix uses the superseded split and does not label actual versus predicted axes explicitly;
- the old category pie chart reports a 63% top-eight share without the revised denominator clarification; and
- the old metric and PR plots do not match the revised 60/20/20 multi-seed protocol.

## Rebuild canonical artifacts

```bash
python src/experiment_v2.py
python src/experiment_synth_v2.py
python src/make_figures_v2.py
python src/validate_release.py
```

Do not copy output numbers manually. Commit the generated per-seed tables, summaries, JSON audit files, and figure manifest.

## Publish

1. Review the generated `outputs/pipeline_b/supplementary_checks.json`.
2. Confirm the manuscript values against the new summaries.
3. Commit all canonical files and record the commit hash.
4. Publish a new GitHub release, recommended tag `v2.0.1`.
5. Create a new Zenodo version from that release.
6. Replace the version DOI, release tag, and commit hash in the manuscript, response letter, and cover letter.
7. Keep the concept DOI unchanged.
