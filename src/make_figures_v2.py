#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the nine main and eight supplementary figures.

The script reads only the archived data snapshot and canonical CSV outputs produced
by ``experiment_v2.py`` and ``experiment_synth_v2.py``.  It does not depend on a
particular user account, home directory, notebook, or container mount point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

import experiment_v2 as pipeline_b


SEED = 101
EXPECTED_MAIN = (
    "Fig1_architecture.png",
    "Fig2_heatmap_weekday_hour.png",
    "Fig3_metrics_bar_ci.png",
    "Fig4_pr_curves.png",
    "Fig5_shap_rf_bar.png",
    "Fig6_threshold_sensitivity.png",
    "Fig7_ablation_pr.png",
    "Fig8_timeout_sweep.png",
    "Fig9_prevalence_sensitivity.png",
)
EXPECTED_SUPPLEMENTARY = (
    "FigS1_cm_IF.png",
    "FigS2_hist_response_time.png",
    "FigS3_category_distribution.png",
    "FigS4_ops_IF.png",
    "FigS5_ops_OCSVM.png",
    "FigS6_ops_RF.png",
    "FigS7_synth_per_type_recall.png",
    "FigS8_synth_f1_vs_rate.png",
)


def repository_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() in {"src", "scripts"} else here


ROOT = repository_root()
DEFAULT_DATA = ROOT / "DATA" / "erm2-nwe9.csv"
DEFAULT_PIPELINE_B = ROOT / "outputs" / "pipeline_b"
DEFAULT_PIPELINE_A = ROOT / "outputs" / "pipeline_a"
DEFAULT_FIGURES = ROOT / "figures_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate all v2 manuscript figures.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pipeline-b-results", type=Path, default=DEFAULT_PIPELINE_B)
    parser.add_argument("--pipeline-a-results", type=Path, default=DEFAULT_PIPELINE_A)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_manifest_path(path: Path) -> str:
    """Return a repository-relative label without exposing a workstation path."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def save_figure(figure: plt.Figure, output_dir: Path, name: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "make_figures_v2.py", "Protocol": "v2"},
    )
    plt.close(figure)
    return path


def prepare_seed_artifacts(cleaned: pd.DataFrame, seed: int) -> dict[str, Any]:
    frame = cleaned.copy()
    frame["is_anomaly"] = (frame["resp_h"] > pipeline_b.TIMEOUT_PRIMARY).astype(int)
    labels = frame["is_anomaly"].to_numpy()
    features_a, categories = pipeline_b.featurize(frame, include_resp=True)
    features_b, _ = pipeline_b.featurize(
        frame, include_resp=False, top_categories=categories
    )
    indices = np.arange(len(frame))
    train, remainder = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        stratify=labels[remainder],
    )

    def fit_if(features: pd.DataFrame) -> tuple[Any, np.ndarray, np.ndarray]:
        model = IsolationForest(random_state=seed, **pipeline_b.IF_PARAMS).fit(
            features.iloc[train]
        )
        return (
            model,
            -model.score_samples(features.iloc[validation]),
            -model.score_samples(features.iloc[test]),
        )

    if_a, if_validation_a, if_test_a = fit_if(features_a)
    _, if_validation_b, if_test_b = fit_if(features_b)
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(train), size=min(15_000, len(train)), replace=False)
    ocsvm = OneClassSVM(**pipeline_b.OCSVM_PARAMS).fit(
        features_a.iloc[train].iloc[subset]
    )
    oc_validation = -ocsvm.score_samples(features_a.iloc[validation])
    oc_test = -ocsvm.score_samples(features_a.iloc[test])
    rf_a = RandomForestClassifier(random_state=seed, **pipeline_b.RF_PARAMS).fit(
        features_a.iloc[train], labels[train]
    )
    rf_b = RandomForestClassifier(random_state=seed, **pipeline_b.RF_PARAMS).fit(
        features_b.iloc[train], labels[train]
    )
    rf_validation_a = rf_a.predict_proba(features_a.iloc[validation])[:, 1]
    rf_test_a = rf_a.predict_proba(features_a.iloc[test])[:, 1]
    rf_test_b = rf_b.predict_proba(features_b.iloc[test])[:, 1]
    return {
        "frame": frame,
        "labels": labels,
        "features_a": features_a,
        "features_b": features_b,
        "train": train,
        "validation": validation,
        "test": test,
        "if_a": if_a,
        "if_validation_a": if_validation_a,
        "if_test_a": if_test_a,
        "if_validation_b": if_validation_b,
        "if_test_b": if_test_b,
        "oc_validation": oc_validation,
        "oc_test": oc_test,
        "rf_a": rf_a,
        "rf_validation_a": rf_validation_a,
        "rf_test_a": rf_test_a,
        "rf_test_b": rf_test_b,
    }


def make_figures(
    data_path: Path,
    pipeline_b_dir: Path,
    pipeline_a_dir: Path,
    output_dir: Path,
    *,
    dpi: int,
    seed: int,
) -> None:
    if dpi < 300:
        raise ValueError("Publication figures must be generated at 300 dpi or higher.")
    data_path = require_file(data_path)
    pipeline_b_dir = pipeline_b_dir.expanduser().resolve()
    pipeline_a_dir = pipeline_a_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_results = pd.read_csv(require_file(pipeline_b_dir / "main_expA_with_resp.csv"))
    timeout_results = pd.read_csv(require_file(pipeline_b_dir / "timeout_sweep.csv"))
    prevalence_results = pd.read_csv(
        require_file(pipeline_b_dir / "prevalence_sensitivity.csv")
    )
    synthetic_results = pd.read_csv(require_file(pipeline_a_dir / "synth_metrics.csv"))
    cleaned, audit = pipeline_b.load_clean(data_path)
    artifacts = prepare_seed_artifacts(cleaned, seed)
    frame = artifacts["frame"]
    labels = artifacts["labels"]
    train = artifacts["train"]
    validation = artifacts["validation"]
    test = artifacts["test"]
    y_validation = labels[validation]
    y_test = labels[test]

    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Figure 1: framework architecture.
    figure, axis = plt.subplots(figsize=(7.0, 1.6))
    boxes = (
        "311 log ingestion\n& transformation",
        "Delay-label\nconstruction",
        "Anomaly detection\n(IF / OCSVM / RF / rule)",
        "Evaluation &\nvisualization",
    )
    for index, text in enumerate(boxes):
        axis.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (index * 1.9, 0),
                1.55,
                0.9,
                boxstyle="round,pad=0.08",
                facecolor="#dbe9f6",
                edgecolor="#3b6ea5",
                linewidth=1.2,
            )
        )
        axis.text(index * 1.9 + 0.78, 0.45, text, ha="center", va="center", fontsize=8.5)
        if index < len(boxes) - 1:
            axis.annotate(
                "",
                xy=(index * 1.9 + 1.83, 0.45),
                xytext=(index * 1.9 + 1.62, 0.45),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#3b6ea5"},
            )
    axis.set_xlim(-0.15, 7.5)
    axis.set_ylim(-0.15, 1.05)
    axis.axis("off")
    save_figure(figure, output_dir, "Fig1_architecture", dpi)

    # Figure 2: workload heatmap.
    workload = cleaned.assign(
        weekday=cleaned["_created"].dt.weekday,
        hour=cleaned["_created"].dt.hour,
    )
    matrix = workload.pivot_table(
        index="weekday", columns="hour", values="resp_h", aggfunc="count"
    ).fillna(0)
    figure, axis = plt.subplots(figsize=(6.4, 3.0))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    figure.colorbar(image, ax=axis, label="Record count")
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Weekday (0 = Monday)")
    axis.set_xticks(range(0, 24, 2))
    axis.set_yticks(range(7))
    axis.set_title("Transformed 311 service-log workload (proxy data)")
    save_figure(figure, output_dir, "Fig2_heatmap_weekday_hour", dpi)

    # Figure 3: mean metrics and t-interval CIs.
    results = main_results.set_index("model")
    models = ("IF", "OCSVM", "RF", "RuleCat80")
    model_labels = (
        "Isolation\nForest",
        "One-Class\nSVM",
        "Random Forest\n(supervised ref.)",
        "Rule baseline\n(cat. p80)",
    )
    metrics = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"))
    x_positions = np.arange(len(models))
    width = 0.25
    figure, axis = plt.subplots(figsize=(6.6, 3.2))
    for index, (metric, label) in enumerate(metrics):
        means = [results.loc[model, f"{metric}_mean"] for model in models]
        errors = [
            results.loc[model, f"{metric}_mean"]
            - results.loc[model, f"{metric}_ci_lo"]
            for model in models
        ]
        axis.bar(
            x_positions + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=label,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(model_labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Detection performance, timeout = 2.0 h (mean ± 95% CI, 10 seeds)")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.3)
    save_figure(figure, output_dir, "Fig3_metrics_bar_ci", dpi)

    # Figure 4: representative PR curves.
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    score_series = (
        ("Isolation Forest", artifacts["if_test_a"]),
        ("One-Class SVM", artifacts["oc_test"]),
        ("Random Forest (sup.)", artifacts["rf_test_a"]),
    )
    for label, scores in score_series:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        axis.plot(
            recall,
            precision,
            label=f"{label} (AP={average_precision_score(y_test, scores):.3f})",
        )
    axis.axhline(
        y_test.mean(),
        linestyle="--",
        color="grey",
        linewidth=1,
        label=f"Chance (prevalence={y_test.mean():.3f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Precision–recall curves, test set (seed {seed})")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.3)
    save_figure(figure, output_dir, "Fig4_pr_curves", dpi)

    # Figure 5: RF SHAP attribution.
    import shap

    features_a = artifacts["features_a"]
    background_pool = features_a.iloc[train].sample(1_000, random_state=seed)
    # TreeExplainer otherwise silently reduces tabular backgrounds to 100 rows.
    # Make that effective sample explicit and reproducible in both code and metadata.
    background = background_pool.sample(100, random_state=seed)
    explained = features_a.iloc[test].sample(2_000, random_state=seed)
    explainer = shap.TreeExplainer(
        artifacts["rf_a"], data=background, model_output="probability"
    )
    shap_values = explainer.shap_values(explained, check_additivity=False)
    if isinstance(shap_values, list):
        class_one_values = shap_values[1]
    else:
        class_one_values = (
            shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
        )
    shap.summary_plot(
        class_one_values,
        explained,
        plot_type="bar",
        show=False,
        max_display=11,
    )
    figure = plt.gcf()
    figure.set_size_inches(7.2, 4.8)
    plt.xlabel(
        "Mean |SHAP value| (contribution to anomaly probability)", fontsize=9
    )
    plt.title("RF feature attribution (supervised reference, v2 protocol)", fontsize=10)
    save_figure(figure, output_dir, "Fig5_shap_rf_bar", dpi)
    with (output_dir / "shap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "shap_version": shap.__version__,
                "explainer": "TreeExplainer",
                "model": "RandomForestClassifier (supervised reference)",
                "background_candidate_pool": "1000 training records (seeded sample)",
                "effective_background": "100 records sampled from the candidate pool",
                "explained": "2000 test records (seeded sample)",
                "output": "probability of anomaly class",
                "seed": seed,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    # Figure 6: test-score threshold sensitivity.
    quantiles = np.linspace(1, 99.5, 120)
    precision_values, recall_values, f1_values, alert_values = [], [], [], []
    for quantile in quantiles:
        threshold = np.percentile(artifacts["if_test_a"], quantile)
        prediction = (artifacts["if_test_a"] >= threshold).astype(int)
        tp = int(((prediction == 1) & (y_test == 1)).sum())
        fp = int(((prediction == 1) & (y_test == 0)).sum())
        fn = int(((prediction == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        alert_values.append(float(prediction.mean()))
    figure, axis = plt.subplots(figsize=(5.8, 3.4))
    axis.plot(quantiles, precision_values, label="Precision")
    axis.plot(quantiles, recall_values, label="Recall")
    axis.plot(quantiles, f1_values, label="F1-score")
    axis.plot(quantiles, alert_values, linestyle=":", color="black", label="Alert rate")
    axis.set_xlabel("Score threshold (percentile of anomaly score)")
    axis.set_ylabel("Value")
    axis.set_ylim(0, 1.02)
    axis.set_title(f"Isolation Forest threshold sensitivity (test set, seed {seed})")
    axis.legend()
    axis.grid(alpha=0.3)
    save_figure(figure, output_dir, "Fig6_threshold_sensitivity", dpi)

    # Figure 7: target-leakage ablation.
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    ablation_scores = (
        ("IF, with resp_h (A)", artifacts["if_test_a"], "-"),
        ("IF, without resp_h (B)", artifacts["if_test_b"], "--"),
        ("RF, with resp_h (A)", artifacts["rf_test_a"], "-"),
        ("RF, without resp_h (B)", artifacts["rf_test_b"], "--"),
    )
    for label, scores, linestyle in ablation_scores:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        axis.plot(
            recall,
            precision,
            linestyle,
            label=f"{label} (AP={average_precision_score(y_test, scores):.3f})",
        )
    axis.axhline(y_test.mean(), linestyle=":", color="grey", linewidth=1)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Feature-ablation (target-leakage) analysis, test set")
    axis.legend(loc="lower left", fontsize=7.5)
    axis.grid(alpha=0.3)
    save_figure(figure, output_dir, "Fig7_ablation_pr", dpi)

    # Figure 8: timeout sensitivity.
    isolation_timeout = timeout_results[timeout_results["model"] == "IF"]
    figure, axis = plt.subplots(figsize=(5.8, 3.4))
    axis.plot(isolation_timeout["timeout_h"], isolation_timeout["f1_mean"], "o-", label="F1-score")
    axis.plot(isolation_timeout["timeout_h"], isolation_timeout["ap_mean"], "s-", label="Average precision")
    axis.plot(isolation_timeout["timeout_h"], isolation_timeout["auc_roc_mean"], "^-", label="AUC-ROC")
    axis.plot(isolation_timeout["timeout_h"], isolation_timeout["prevalence"], "d:", color="grey", label="Label prevalence")
    axis.set_xscale("log")
    axis.set_xticks(isolation_timeout["timeout_h"])
    axis.set_xticklabels([f"{value:g}" for value in isolation_timeout["timeout_h"]])
    axis.set_xlabel("Delay-label timeout (hours, log scale)")
    axis.set_ylabel("Value")
    axis.set_ylim(0, 1.02)
    axis.set_title("Sensitivity to delay-label definition (IF, 5 seeds)")
    axis.legend()
    axis.grid(alpha=0.3)
    save_figure(figure, output_dir, "Fig8_timeout_sweep", dpi)

    # Figure 9: low-prevalence retraining analysis.
    isolation_prevalence = prevalence_results[prevalence_results["model"] == "IF"]
    figure, axis = plt.subplots(figsize=(5.8, 3.4))
    for metric, label, marker in (
        ("precision", "Precision", "o"),
        ("recall", "Recall", "s"),
        ("f1", "F1-score", "^"),
        ("auc_roc", "AUC-ROC", "d"),
    ):
        mean = isolation_prevalence[f"{metric}_mean"]
        axis.errorbar(
            isolation_prevalence["prevalence_target"] * 100,
            mean,
            yerr=(
                mean - isolation_prevalence[f"{metric}_ci_lo"],
                isolation_prevalence[f"{metric}_ci_hi"] - mean,
            ),
            marker=marker,
            capsize=3,
            label=label,
        )
    axis.set_xlabel("Anomaly prevalence (%) — downsampled positives, timeout = 2.0 h")
    axis.set_ylabel("Value")
    axis.set_ylim(0, 1.02)
    axis.set_title("Low-prevalence operating regime (IF, 5 seeds, mean ± 95% CI)")
    axis.legend()
    axis.grid(alpha=0.3)
    save_figure(figure, output_dir, "Fig9_prevalence_sensitivity", dpi)

    # Supplementary Figure S1: confusion matrix with explicit directions.
    threshold = pipeline_b.pick_threshold_val(artifacts["if_validation_a"], y_validation)
    matrix = confusion_matrix(y_test, artifacts["if_test_a"] >= threshold)
    figure, axis = plt.subplots(figsize=(3.4, 3.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set_xticks((0, 1), labels=("Normal", "Anomaly"))
    axis.set_yticks((0, 1), labels=("Normal", "Anomaly"))
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(f"Isolation Forest confusion matrix\n(test set, seed {seed})")
    save_figure(figure, output_dir, "FigS1_cm_IF", dpi)

    # Supplementary Figure S2: cleaned response-time distribution.
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    axis.hist(cleaned["resp_h"], bins=80)
    axis.set_yscale("log")
    axis.set_xlabel("Response time (hours)")
    axis.set_ylabel("Count (log scale)")
    axis.set_title(f"Response-time distribution after cleaning (n = {len(cleaned):,})")
    axis.annotate(
        f"Zero-duration closures: {audit['n_zero']:,} (retained)\n"
        f"Missing closure: {audit['n_missing_closure']:,} (excluded)\n"
        f"Negative intervals: {audit['n_negative']:,} (excluded)",
        xy=(0.45, 0.72),
        xycoords="axes fraction",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "grey", "alpha": 0.9},
    )
    save_figure(figure, output_dir, "FigS2_hist_response_time", dpi)

    # Supplementary Figure S3: honestly labeled 311 source categories.
    category_counts = cleaned["_category"].value_counts()
    top = category_counts.head(8)
    labels_s3 = list(top.index) + ["Other"]
    percentages = list(top / len(cleaned) * 100) + [category_counts.iloc[8:].sum() / len(cleaned) * 100]
    figure, axis = plt.subplots(figsize=(5.8, 3.0))
    axis.bar(labels_s3, percentages, color="#4878b0")
    axis.set_ylabel("Share of retained records (%)")
    axis.set_title("311 source service categories (agency codes; not clinical categories)")
    axis.tick_params(axis="x", rotation=30)
    for index, value in enumerate(percentages):
        axis.text(index, value + 0.5, f"{value:.1f}", ha="center", fontsize=7.5)
    save_figure(figure, output_dir, "FigS3_category_distribution", dpi)

    # Supplementary Figures S4-S6: per-model operating trade-offs.
    for label, scores, filename in (
        ("Isolation Forest", artifacts["if_test_a"], "FigS4_ops_IF"),
        ("One-Class SVM", artifacts["oc_test"], "FigS5_ops_OCSVM"),
        ("Random Forest (supervised ref.)", artifacts["rf_test_a"], "FigS6_ops_RF"),
    ):
        recalls, precisions = [], []
        for quantile in np.linspace(50, 99.5, 80):
            prediction = (scores >= np.percentile(scores, quantile)).astype(int)
            tp = int(((prediction == 1) & (y_test == 1)).sum())
            fp = int(((prediction == 1) & (y_test == 0)).sum())
            fn = int(((prediction == 0) & (y_test == 1)).sum())
            precisions.append(tp / (tp + fp) if tp + fp else 1.0)
            recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        figure, axis = plt.subplots(figsize=(4.6, 3.2))
        axis.plot(recalls, precisions)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title(f"Operational trade-off — {label}\n(threshold percentiles 50–99.5, test set)")
        axis.grid(alpha=0.3)
        save_figure(figure, output_dir, filename, dpi)

    # Supplementary Figure S7: per-type recall at 10% nominal injection.
    synthetic_ten = synthetic_results[
        synthetic_results["nominal_injection_rate"] == 0.10
    ].set_index("model")
    anomaly_types = ("delay", "repeat", "volume", "timestamp")
    models_s7 = ("IF", "OCSVM", "RF", "Rule60s")
    labels_s7 = ("IF", "OCSVM", "RF (sup.)", "Rule (>60 s)")
    x_positions = np.arange(len(anomaly_types))
    width = 0.2
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    for index, (model, label) in enumerate(zip(models_s7, labels_s7)):
        values = [synthetic_ten.loc[model, f"recall_{name}_mean"] for name in anomaly_types]
        axis.bar(x_positions + (index - 1.5) * width, values, width, label=label)
    axis.set_xticks(x_positions, labels=("Delayed\nresponse", "Repeated\ncall", "Volume\nburst", "Invalid\ntimestamp"))
    axis.set_ylabel("Recall by injected anomaly type")
    axis.set_ylim(0, 1.05)
    axis.set_title("Synthetic-injection pipeline: per-type recall (AR = 0.10, 10 seeds)")
    axis.legend(ncols=4, fontsize=7.5)
    axis.grid(axis="y", alpha=0.3)
    save_figure(figure, output_dir, "FigS7_synth_per_type_recall", dpi)

    # Supplementary Figure S8: F1 across nominal injection rates.
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    for model, label, marker in (("IF", "IF", "o"), ("OCSVM", "OCSVM", "s"), ("RF", "RF (sup.)", "^")):
        selected = synthetic_results[synthetic_results["model"] == model]
        mean = selected["f1_mean"]
        axis.errorbar(
            selected["nominal_injection_rate"] * 100,
            mean,
            yerr=(mean - selected["f1_ci_lo"], selected["f1_ci_hi"] - mean),
            marker=marker,
            capsize=3,
            label=label,
        )
    axis.set_xlabel("Injection rate (%)  [effective prevalence reported in run_meta.json]")
    axis.set_ylabel("F1-score")
    axis.set_ylim(0, 1.0)
    axis.set_title("Synthetic-injection pipeline: F1 vs injection rate (mean ± 95% CI)")
    axis.legend()
    axis.grid(alpha=0.3)
    save_figure(figure, output_dir, "FigS8_synth_f1_vs_rate", dpi)

    generated = sorted(path.name for path in output_dir.glob("Fig*.png"))
    expected = sorted((*EXPECTED_MAIN, *EXPECTED_SUPPLEMENTARY))
    if generated != expected:
        missing = sorted(set(expected) - set(generated))
        unexpected = sorted(set(generated) - set(expected))
        raise RuntimeError(f"Figure-set mismatch. Missing={missing}; unexpected={unexpected}")
    manifest = {
        "protocol": "v2",
        "dpi": dpi,
        "seed_for_representative_figures": seed,
        "data_sha256": sha256_file(data_path),
        "pipeline_b_results": portable_manifest_path(pipeline_b_dir),
        "pipeline_a_results": portable_manifest_path(pipeline_a_dir),
        "figures": [
            {"file": name, "sha256": sha256_file(output_dir / name)} for name in generated
        ],
    }
    with (output_dir / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Generated {len(generated)} figures in {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    make_figures(
        args.data,
        args.pipeline_b_results,
        args.pipeline_a_results,
        args.output_dir,
        dpi=args.dpi,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
