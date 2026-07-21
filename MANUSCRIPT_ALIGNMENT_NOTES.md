# Mandatory manuscript alignment before submission

The code and archived outputs must describe the same protocol. Complete the
following checks after the final Docker rerun and before creating the Zenodo
version.

## 1. Correct the SHAP background description

The plotting script now makes SHAP's effective background size explicit. It
draws a seeded 1,000-record candidate pool and then selects 100 records as the
effective `TreeExplainer` background. This avoids SHAP's otherwise silent
tabular-background reduction and records both quantities in
`figures_v2/shap_metadata.json`.

Replace the manuscript's current "1,000-record background" statement with:

> SHAP feature attribution used `TreeExplainer` (SHAP v0.52.0) applied directly
> to the supervised random-forest reference, with a seeded candidate pool of
> 1,000 training records from which 100 records were selected as the effective
> background, 2,000 seeded test records explained, and probability-scale model
> output.

Apply the same correction to the response-to-reviewers document.

## 2. Correct the rule-baseline interpretation

At the 10% nominal injection rate, the 60-s rule has mean per-type recall of
1.000 for delayed responses, 0.071 for repeated calls, 0.041 for volume bursts,
and 0.059 for invalid timestamps. Therefore, the sentence "The 60-s rule
detected only delays" is too absolute.

Use:

> The 60-s rule recovered delayed responses perfectly and flagged only small
> fractions of the other observable injected types (mean recall 0.071 for
> repeats, 0.041 for bursts, and 0.059 for invalid timestamps), so it functioned
> primarily as a delay detector.

## 3. Reconcile exact paired-test values

The source now archives both manuscript-reported paired comparisons:

- Isolation Forest F1, with versus without `resp_h`;
- Isolation Forest average precision, with versus without `resp_h`.

The supplied QA outputs were generated in the available local environment and
give F1 `t=38.656`, `p=2.58e-11` and AP `t=41.179`, `p=1.46e-11`. The current
manuscript reports nearby values from its declared pinned environment. Run
`docker compose run --rm reproduce`, then copy the exact values from
`outputs/pipeline_b/statistical_comparisons.json` into the abstract, Results,
cover letter, and response letter. Do not mix statistics across environments.

## 4. Reconcile runtime benchmarks

Runtime values depend on the CPU quota and host. The benchmark code fixes both
estimators to one model job, uses 70,000 training events, and records the cgroup
CPU quota, host logical CPU count, Python version, throughput, latency, model
size, and peak process memory. Retain the manuscript's existing runtime numbers
only if a final one-CPU Docker run reproduces them; otherwise update every
occurrence consistently and state the measured environment.

## 5. Publish a new immutable version

The current v2.0.0 Zenodo record cannot be overwritten. After the final pinned
rerun, commit the complete canonical package, publish a new GitHub release
(recommended `v2.0.1`), create a new Zenodo version, and replace the version DOI,
release tag, and commit hash everywhere in the submission package. Keep the
concept DOI unchanged.
