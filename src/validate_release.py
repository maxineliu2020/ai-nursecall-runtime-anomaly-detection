#!/usr/bin/env python3
"""Fail-fast validation for a GitHub/Zenodo reproducibility release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence


EXPECTED_DATA_SHA256 = "bf29e88716efe5cb4d6fd26e3fc4d46edf05f6545ea3481e99c46199197b97c0"
EXPECTED_FIGURES = {
    *(f"Fig{number}_{suffix}.png" for number, suffix in (
        (1, "architecture"),
        (2, "heatmap_weekday_hour"),
        (3, "metrics_bar_ci"),
        (4, "pr_curves"),
        (5, "shap_rf_bar"),
        (6, "threshold_sensitivity"),
        (7, "ablation_pr"),
        (8, "timeout_sweep"),
        (9, "prevalence_sensitivity"),
    )),
    "FigS1_cm_IF.png",
    "FigS2_hist_response_time.png",
    "FigS3_category_distribution.png",
    "FigS4_ops_IF.png",
    "FigS5_ops_OCSVM.png",
    "FigS6_ops_RF.png",
    "FigS7_synth_per_type_recall.png",
    "FigS8_synth_f1_vs_rate.png",
}
REQUIRED_PIPELINE_B_OUTPUTS = {
    "data_audit.json",
    "environment.json",
    "main_expA_with_resp.csv",
    "main_expA_with_resp_runs.csv",
    "main_expB_no_resp.csv",
    "main_expB_no_resp_runs.csv",
    "timeout_sweep.csv",
    "prevalence_sensitivity.csv",
    "prevalence_sensitivity_runs.csv",
    "prevalence_1pct_no_resp.csv",
    "prevalence_1pct_no_resp_runs.csv",
    "fixed_model_control.csv",
    "fixed_model_control_runs.csv",
    "split_meta.json",
    "statistical_comparisons.json",
    "supplementary_checks.json",
}
REQUIRED_PIPELINE_A_OUTPUTS = {
    "environment.json",
    "run_meta.json",
    "synth_metrics.csv",
    "synth_metrics_runs.csv",
}
FORBIDDEN_PATHS = re.compile(
    "|".join(
        (
            "/" + "home" + "/",
            "/" + "mnt" + "/",
            r"[A-Za-z]:\\" + "Users" + r"\\",
        )
    )
)


def root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(directory: Path, expected: set[str], label: str) -> list[str]:
    missing = sorted(name for name in expected if not (directory / name).is_file())
    return [f"{label}: missing {name}" for name in missing]


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    data = root / "DATA" / "erm2-nwe9.csv"
    if not data.is_file():
        errors.append("data: missing DATA/erm2-nwe9.csv")
    elif sha256_file(data) != EXPECTED_DATA_SHA256:
        errors.append("data: SHA-256 does not match the manuscript snapshot")

    errors.extend(
        require_files(
            root / "outputs" / "pipeline_b",
            REQUIRED_PIPELINE_B_OUTPUTS,
            "Pipeline B",
        )
    )
    errors.extend(
        require_files(
            root / "outputs" / "pipeline_a",
            REQUIRED_PIPELINE_A_OUTPUTS,
            "Pipeline A",
        )
    )
    errors.extend(require_files(root / "figures_v2", EXPECTED_FIGURES, "figures"))

    checks_path = root / "outputs" / "pipeline_b" / "supplementary_checks.json"
    if checks_path.is_file():
        try:
            checks = json.loads(checks_path.read_text(encoding="utf-8"))
            if checks.get("data_sha256") != EXPECTED_DATA_SHA256:
                errors.append("supplementary checks: data SHA-256 mismatch")
            for key in ("fixed_model_control", "leakage_free_1pct_retraining"):
                if key not in checks:
                    errors.append(f"supplementary checks: missing section {key}")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"supplementary checks: invalid JSON ({error})")

    comparisons_path = root / "outputs" / "pipeline_b" / "statistical_comparisons.json"
    if comparisons_path.is_file():
        try:
            comparisons = json.loads(comparisons_path.read_text(encoding="utf-8"))
            expected = {
                "IF_F1_with_vs_without_resp_h",
                "IF_AP_with_vs_without_resp_h",
            }
            missing = sorted(expected - set(comparisons))
            if missing:
                errors.append(
                    "statistical comparisons: missing " + ", ".join(missing)
                )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"statistical comparisons: invalid JSON ({error})")

    figure_manifest = root / "figures_v2" / "figure_manifest.json"
    if not figure_manifest.is_file():
        errors.append("figures: missing figure_manifest.json")
    else:
        try:
            manifest = json.loads(figure_manifest.read_text(encoding="utf-8"))
            listed = {entry["file"] for entry in manifest.get("figures", [])}
            if listed != EXPECTED_FIGURES:
                errors.append("figures: manifest does not list exactly 17 expected figures")
            if manifest.get("data_sha256") != EXPECTED_DATA_SHA256:
                errors.append("figures: manifest data SHA-256 mismatch")
            for key in ("pipeline_a_results", "pipeline_b_results"):
                label = manifest.get(key, "")
                if not label or Path(label).is_absolute():
                    errors.append(f"figures: {key} must be repository-relative")
            for entry in manifest.get("figures", []):
                figure_path = root / "figures_v2" / entry["file"]
                if figure_path.is_file() and sha256_file(figure_path) != entry.get("sha256"):
                    errors.append(f"figures: checksum mismatch for {entry['file']}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            errors.append(f"figures: invalid figure manifest ({error})")

    shap_metadata = root / "figures_v2" / "shap_metadata.json"
    if not shap_metadata.is_file():
        errors.append("figures: missing shap_metadata.json")
    else:
        try:
            metadata = json.loads(shap_metadata.read_text(encoding="utf-8"))
            for key in ("explainer", "effective_background", "explained", "output"):
                if not metadata.get(key):
                    errors.append(f"SHAP metadata: missing {key}")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"SHAP metadata: invalid JSON ({error})")

    for directory in (root / "src", root / "streamlit"):
        if not directory.is_dir():
            errors.append(f"source: missing directory {directory.relative_to(root)}")
            continue
        for path in directory.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN_PATHS.search(text):
                errors.append(
                    f"source: machine-specific absolute path in {path.relative_to(root)}"
                )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_from_script())
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    errors = validate(root)
    if errors:
        print("Release validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Release validation passed: data, code, outputs, and 17 figures are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
