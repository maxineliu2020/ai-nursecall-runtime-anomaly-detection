#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline B: reproducible NYC 311 proxy-data experiments.

This module implements the complete protocol reported in the revised manuscript:

* documented cleaning of missing, negative, zero, and extreme response intervals;
* stratified 60/20/20 train/validation/test splits;
* validation-only operating-threshold selection;
* ten-seed primary analysis and target-leakage ablation;
* timeout and low-prevalence sensitivity analyses;
* a fixed-model base-rate control;
* a 1% prevalence leakage-free retraining analysis;
* positive-subsample composition checks; and
* runtime and environment reporting.

All paths are configurable.  By default, data are read from ``DATA/`` and outputs
are written below ``outputs/pipeline_b/`` relative to the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on native Windows
    resource = None


PRIMARY_SEEDS = tuple(range(101, 111))
TIMEOUT_PRIMARY = 2.0
TIMEOUT_SWEEP = (1.0, 2.0, 6.0, 12.0, 24.0, 48.0)
PREVALENCE_TARGETS = (0.01, 0.02, 0.05, 0.10)
N_SENSITIVITY_SEEDS = 5
N_FIXED_CONTROL_SEEDS = 3

# One worker matches the single-vCPU canonical benchmark and prevents host-dependent
# parallelism from changing the reported timing conditions.
IF_PARAMS = {"n_estimators": 400, "contamination": "auto", "n_jobs": 1}
RF_PARAMS = {"n_estimators": 400, "n_jobs": 1}
OCSVM_PARAMS = {"kernel": "rbf", "gamma": "scale", "nu": 0.05}


def repository_root() -> Path:
    """Return the repository root for scripts stored in ``src/`` or the root."""
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() in {"src", "scripts"} else here


REPOSITORY_ROOT = repository_root()
DEFAULT_DATA = REPOSITORY_ROOT / "DATA" / "erm2-nwe9.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "pipeline_b"


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one integer seed is required.")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete Pipeline B reproducibility protocol."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Input CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=PRIMARY_SEEDS,
        help="Comma-separated primary seeds (default: 101,...,110).",
    )
    parser.add_argument(
        "--skip-ocsvm",
        action="store_true",
        help="Skip OCSVM in the primary run (not for canonical manuscript outputs).",
    )
    parser.add_argument(
        "--skip-runtime-benchmark",
        action="store_true",
        help="Skip hardware-dependent runtime benchmarking.",
    )
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"Input data are missing required columns: {', '.join(missing)}")


def load_clean(data_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the archived snapshot and apply the manuscript cleaning rules."""
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Input data not found: {data_path}. Pass --data or place the snapshot "
            "at DATA/erm2-nwe9.csv."
        )

    frame = pd.read_csv(data_path, low_memory=False)
    require_columns(frame, ("created_date", "closed_date", "agency"))
    n_raw = len(frame)
    frame["_created"] = pd.to_datetime(frame["created_date"], errors="coerce")
    frame["_closed"] = pd.to_datetime(frame["closed_date"], errors="coerce")
    frame["_category"] = frame["agency"].astype("string").fillna("Unknown")
    frame["resp_h"] = (
        frame["_closed"] - frame["_created"]
    ).dt.total_seconds() / 3600.0

    missing = frame["resp_h"].isna()
    negative = frame["resp_h"] < 0
    zero = frame["resp_h"] == 0
    over_seven_days = frame["resp_h"] > 168
    retained = frame.loc[~missing & ~negative].copy()
    retained = retained.sort_values("_created").reset_index(drop=True)

    if retained.empty:
        raise ValueError("No records remain after cleaning.")

    audit = {
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "n_raw": n_raw,
        "n_missing_closure": int(missing.sum()),
        "n_negative": int(negative.sum()),
        "n_zero": int(zero.sum()),
        "n_gt_7d": int(over_seven_days.sum()),
        "n_retained": len(retained),
        "created_min": frame["_created"].min(),
        "created_max": frame["_created"].max(),
    }
    return retained, audit


def featurize(
    frame: pd.DataFrame,
    *,
    include_resp: bool,
    top_categories: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct the manuscript feature matrix."""
    features = pd.DataFrame(index=frame.index)
    features["hour"] = frame["_created"].dt.hour
    features["weekday"] = frame["_created"].dt.weekday
    if include_resp:
        features["resp_h"] = frame["resp_h"]

    if top_categories is None:
        top_categories = frame["_category"].value_counts().head(8).index.tolist()
    categories = [str(category) for category in top_categories]
    for category in categories:
        features[f"cat_{category}"] = (frame["_category"] == category).astype(int)
    return features.fillna(0), categories


def label_frame(frame: pd.DataFrame, timeout_h: float) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["is_anomaly"] = (labeled["resp_h"] > timeout_h).astype(int)
    if labeled["is_anomaly"].nunique() != 2:
        raise ValueError(f"Timeout {timeout_h:g} h does not produce two label classes.")
    return labeled


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "median_h": float(np.median(array)),
        "p90_h": float(np.percentile(array, 90)),
    }


def sample_to_prevalence(
    labeled: pd.DataFrame,
    target: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Downsample positives uniformly without replacement; retain all negatives."""
    if not 0 < target < 1:
        raise ValueError("Prevalence target must be between 0 and 1.")

    positive_indices = labeled.index[labeled["is_anomaly"] == 1].to_numpy()
    negative_indices = labeled.index[labeled["is_anomaly"] == 0].to_numpy()
    n_positive_keep = int(round(target * len(negative_indices) / (1 - target)))
    if n_positive_keep < 2:
        raise ValueError("Prevalence target leaves too few positive records.")
    n_positive_keep = min(n_positive_keep, len(positive_indices))

    rng = np.random.default_rng(seed)
    sampled_positive = rng.choice(
        positive_indices, size=n_positive_keep, replace=False
    )
    kept_indices = np.sort(np.concatenate((sampled_positive, negative_indices)))
    sampled = labeled.loc[kept_indices].reset_index(drop=True)

    metadata = {
        "seed": seed,
        "target_prevalence": target,
        "actual_prevalence": float(sampled["is_anomaly"].mean()),
        "n_total": len(sampled),
        "n_negative_retained": len(negative_indices),
        "n_positive_retained": n_positive_keep,
        "full_positive_distribution": distribution_summary(
            labeled.loc[positive_indices, "resp_h"]
        ),
        "sampled_positive_distribution": distribution_summary(
            labeled.loc[sampled_positive, "resp_h"]
        ),
    }
    return sampled, metadata


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    indices = np.arange(len(labels))
    train, remainder = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        stratify=labels[remainder],
    )
    return train, validation, test


def full_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    hours_span: float,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "alert_rate": float(np.mean(predictions)),
        "alerts_per_hour": float(np.sum(predictions)) / hours_span
        if hours_span > 0
        else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if scores is not None and np.unique(labels).size > 1:
        result["ap"] = average_precision_score(labels, scores)
        result["auc_roc"] = roc_auc_score(labels, scores)
    return result


def pick_threshold_val(scores: np.ndarray, labels: np.ndarray) -> float:
    """Select the operating threshold by maximum F1 on validation data only."""
    candidates = np.unique(np.percentile(scores, np.linspace(1, 99.5, 300)))
    if candidates.size == 0:
        raise ValueError("No threshold candidates were produced.")
    f1_values = [
        f1_score(labels, scores >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(f1_values))])


def mean_ci(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size > 1:
        half_width = float(
            stats.t.ppf(0.975, array.size - 1) * stats.sem(array)
        )
        standard_deviation = float(array.std(ddof=1))
    else:
        half_width = float("nan")
        standard_deviation = 0.0
    return mean, mean - half_width, mean + half_width, standard_deviation


def aggregate(nested_rows: Sequence[dict[str, dict[str, Any]]]) -> pd.DataFrame:
    models = sorted({model for row in nested_rows for model in row})
    records: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row[model] for row in nested_rows if model in row]
        record: dict[str, Any] = {"model": model, "n_runs": len(model_rows)}
        metric_names = sorted(
            {
                name
                for row in model_rows
                for name, value in row.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for name in metric_names:
            values = [float(row[name]) for row in model_rows if np.isfinite(row[name])]
            if not values:
                continue
            mean, low, high, standard_deviation = mean_ci(values)
            record[f"{name}_mean"] = mean
            record[f"{name}_sd"] = standard_deviation
            record[f"{name}_ci_lo"] = low
            record[f"{name}_ci_hi"] = high
        records.append(record)
    return pd.DataFrame.from_records(records)


def flatten_runs(
    nested_rows: Sequence[dict[str, dict[str, Any]]],
    seeds: Sequence[int],
    **conditions: Any,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for seed, result in zip(seeds, nested_rows):
        for model, metrics in result.items():
            records.append({"seed": seed, "model": model, **conditions, **metrics})
    return pd.DataFrame.from_records(records)


def one_run(
    cleaned: pd.DataFrame,
    timeout_h: float,
    seed: int,
    *,
    include_resp: bool = True,
    prevalence_target: float | None = None,
    run_ocsvm: bool = True,
    ocsvm_cap: int = 15_000,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    frame = label_frame(cleaned, timeout_h)
    sampling_metadata = None
    if prevalence_target is not None:
        frame, sampling_metadata = sample_to_prevalence(
            frame, prevalence_target, seed
        )

    labels = frame["is_anomaly"].to_numpy()
    features, top_categories = featurize(frame, include_resp=include_resp)
    train, validation, test = split_indices(labels, seed)
    x_train = features.iloc[train]
    x_validation = features.iloc[validation]
    x_test = features.iloc[test]
    y_train = labels[train]
    y_validation = labels[validation]
    y_test = labels[test]
    test_created = frame["_created"].iloc[test]
    hours_span = (
        test_created.max() - test_created.min()
    ).total_seconds() / 3600.0

    result: dict[str, dict[str, Any]] = {}

    isolation_forest = IsolationForest(random_state=seed, **IF_PARAMS).fit(x_train)
    if_validation = -isolation_forest.score_samples(x_validation)
    if_test = -isolation_forest.score_samples(x_test)
    if_threshold = pick_threshold_val(if_validation, y_validation)
    result["IF"] = full_metrics(
        y_test, (if_test >= if_threshold).astype(int), if_test, hours_span
    )
    result["IF"]["threshold"] = if_threshold

    training_scores = -isolation_forest.score_samples(x_train)
    for quantile in (0.90, 0.80):
        threshold = float(np.quantile(training_scores, quantile))
        name = f"IF_budget_top{int(round((1 - quantile) * 100))}pct"
        result[name] = full_metrics(
            y_test, (if_test >= threshold).astype(int), None, hours_span
        )
        result[name]["threshold"] = threshold

    if run_ocsvm:
        rng = np.random.default_rng(seed)
        subset = rng.choice(
            len(x_train), size=min(ocsvm_cap, len(x_train)), replace=False
        )
        ocsvm = OneClassSVM(**OCSVM_PARAMS).fit(x_train.iloc[subset])
        oc_validation = -ocsvm.score_samples(x_validation)
        oc_test = -ocsvm.score_samples(x_test)
        oc_threshold = pick_threshold_val(oc_validation, y_validation)
        result["OCSVM"] = full_metrics(
            y_test, (oc_test >= oc_threshold).astype(int), oc_test, hours_span
        )
        result["OCSVM"]["threshold"] = oc_threshold

    random_forest = RandomForestClassifier(
        random_state=seed, **RF_PARAMS
    ).fit(x_train, y_train)
    rf_validation = random_forest.predict_proba(x_validation)[:, 1]
    rf_test = random_forest.predict_proba(x_test)[:, 1]
    rf_threshold = pick_threshold_val(rf_validation, y_validation)
    result["RF"] = full_metrics(
        y_test, (rf_test >= rf_threshold).astype(int), rf_test, hours_span
    )
    result["RF"]["threshold"] = rf_threshold

    if include_resp:
        category_thresholds = (
            frame.iloc[train].groupby("_category")["resp_h"].quantile(0.80)
        )
        global_threshold = float(frame.iloc[train]["resp_h"].quantile(0.80))
        test_thresholds = (
            frame.iloc[test]["_category"]
            .map(category_thresholds)
            .fillna(global_threshold)
            .to_numpy()
        )
        predictions = (
            frame.iloc[test]["resp_h"].to_numpy() > test_thresholds
        ).astype(int)
        result["RuleCat80"] = full_metrics(
            y_test, predictions, None, hours_span
        )

    metadata = {
        "seed": seed,
        "timeout_h": timeout_h,
        "include_resp_h": include_resp,
        "n": len(frame),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "n_test_positive": int(y_test.sum()),
        "prevalence": float(labels.mean()),
        "top_categories": top_categories,
        "sampling": sampling_metadata,
    }
    return result, metadata


def fixed_model_control(
    cleaned: pd.DataFrame,
    seeds: Sequence[int],
    prevalence_targets: Sequence[float] = PREVALENCE_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze model and threshold; downsample test positives only."""
    frame = label_frame(cleaned, TIMEOUT_PRIMARY)
    features, _ = featurize(frame, include_resp=True)
    labels = frame["is_anomaly"].to_numpy()
    records: list[dict[str, Any]] = []

    for seed in seeds:
        train, validation, test = split_indices(labels, seed)
        model = IsolationForest(random_state=seed, **IF_PARAMS).fit(
            features.iloc[train]
        )
        validation_scores = -model.score_samples(features.iloc[validation])
        test_scores = -model.score_samples(features.iloc[test])
        threshold = pick_threshold_val(validation_scores, labels[validation])

        test_labels = labels[test]
        positive = np.flatnonzero(test_labels == 1)
        negative = np.flatnonzero(test_labels == 0)
        test_created = frame["_created"].iloc[test]
        rng = np.random.default_rng(seed)

        conditions: list[tuple[str, float | None]] = [("full", None)]
        conditions.extend((f"{target:.2f}", target) for target in prevalence_targets)
        for condition, target in conditions:
            if target is None:
                selected = np.arange(len(test))
            else:
                n_positive_keep = int(
                    round(target * len(negative) / (1 - target))
                )
                n_positive_keep = min(n_positive_keep, len(positive))
                sampled_positive = rng.choice(
                    positive, size=n_positive_keep, replace=False
                )
                selected = np.sort(np.concatenate((negative, sampled_positive)))

            selected_times = test_created.iloc[selected]
            hours_span = (
                selected_times.max() - selected_times.min()
            ).total_seconds() / 3600.0
            metrics = full_metrics(
                test_labels[selected],
                (test_scores[selected] >= threshold).astype(int),
                test_scores[selected],
                hours_span,
            )
            records.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "target_prevalence": target,
                    "actual_prevalence": float(test_labels[selected].mean()),
                    "n_test": len(selected),
                    "n_test_positive": int(test_labels[selected].sum()),
                    "threshold": threshold,
                    **metrics,
                }
            )

    runs = pd.DataFrame.from_records(records)
    summary_records: list[dict[str, Any]] = []
    for condition, group in runs.groupby("condition", sort=False):
        record: dict[str, Any] = {
            "condition": condition,
            "n_runs": len(group),
            "target_prevalence": group["target_prevalence"].iloc[0],
        }
        for metric in (
            "actual_prevalence",
            "precision",
            "recall",
            "f1",
            "ap",
            "auc_roc",
            "fpr",
            "alert_rate",
        ):
            mean, low, high, standard_deviation = mean_ci(group[metric])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = standard_deviation
            record[f"{metric}_ci_lo"] = low
            record[f"{metric}_ci_hi"] = high
        summary_records.append(record)
    return runs, pd.DataFrame.from_records(summary_records)


def runtime_benchmark(
    cleaned: pd.DataFrame, seed: int = 101, repetitions: int = 200
) -> dict[str, Any]:
    frame = label_frame(cleaned, TIMEOUT_PRIMARY)
    features, _ = featurize(frame, include_resp=True)
    labels = frame["is_anomaly"].to_numpy()
    train_size = min(70_000, len(frame) - 1_000)
    train, test = train_test_split(
        np.arange(len(frame)),
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )
    x_train, x_test = features.iloc[train], features.iloc[test]
    output: dict[str, Any] = {}

    estimators = {
        "IF": IsolationForest(random_state=seed, **IF_PARAMS),
        "RF": RandomForestClassifier(random_state=seed, **RF_PARAMS),
    }
    for name, estimator in estimators.items():
        start = time.perf_counter()
        if name == "RF":
            estimator.fit(x_train, labels[train])
            score = lambda data: estimator.predict_proba(data)[:, 1]
        else:
            estimator.fit(x_train)
            score = lambda data: -estimator.score_samples(data)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        score(x_test)
        batch_seconds = time.perf_counter() - start

        one_record = x_test.iloc[[0]]
        latencies = []
        for _ in range(repetitions):
            start = time.perf_counter()
            score(one_record)
            latencies.append(time.perf_counter() - start)
        latency_ms = np.asarray(latencies) * 1000.0
        output[name] = {
            "fit_s": fit_seconds,
            "train_events": len(x_train),
            "batch_events": len(x_test),
            "batch_s": batch_seconds,
            "events_per_s": len(x_test) / batch_seconds,
            "single_latency_ms_mean": float(latency_ms.mean()),
            "single_latency_ms_sd": float(latency_ms.std(ddof=1)),
            "single_latency_ms_p95": float(np.percentile(latency_ms, 95)),
            "model_size_mb": len(pickle.dumps(estimator)) / 1e6,
        }

    output["hardware"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host_logical_cpu_count": os.cpu_count(),
        "model_n_jobs": 1,
        "cgroup_cpu_limit": cgroup_cpu_limit(),
        "python": sys.version.split()[0],
        "process_peak_rss_mb": None,
    }
    if resource is not None:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output["hardware"]["process_peak_rss_mb"] = (
            peak_rss / 1_000_000.0 if platform.system() == "Darwin" else peak_rss / 1024.0
        )
    return output


def cgroup_cpu_limit() -> float | None:
    """Return the Linux cgroup CPU quota when one is configured."""
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text(encoding="utf-8").split()
        if quota_text != "max":
            return float(quota_text) / float(period_text)
    except (OSError, ValueError):
        pass

    legacy_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    legacy_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota = float(legacy_quota.read_text(encoding="utf-8"))
        period = float(legacy_period.read_text(encoding="utf-8"))
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn_version,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
        "platform": platform.platform(),
    }
    if metadata["scipy"] is None:
        import scipy

        metadata["scipy"] = scipy.__version__
    return metadata


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def run_protocol(
    data_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    *,
    run_ocsvm: bool = True,
    run_benchmark: bool = True,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned, audit = load_clean(data_path)
    write_json(output_dir / "data_audit.json", audit)
    write_json(output_dir / "environment.json", environment_metadata())
    print(f"Retained {len(cleaned):,} of {audit['n_raw']:,} records.")

    primary_with_response: list[dict[str, dict[str, Any]]] = []
    primary_without_response: list[dict[str, dict[str, Any]]] = []
    split_metadata: list[dict[str, Any]] = []
    for seed in seeds:
        with_response, metadata = one_run(
            cleaned,
            TIMEOUT_PRIMARY,
            seed,
            include_resp=True,
            run_ocsvm=run_ocsvm,
        )
        without_response, _ = one_run(
            cleaned,
            TIMEOUT_PRIMARY,
            seed,
            include_resp=False,
            run_ocsvm=False,
        )
        primary_with_response.append(with_response)
        primary_without_response.append(without_response)
        split_metadata.append(metadata)
        print(
            f"seed {seed}: IF with resp_h F1={with_response['IF']['f1']:.3f}; "
            f"without resp_h F1={without_response['IF']['f1']:.3f}"
        )

    write_csv(
        aggregate(primary_with_response), output_dir / "main_expA_with_resp.csv"
    )
    write_csv(
        aggregate(primary_without_response), output_dir / "main_expB_no_resp.csv"
    )
    write_csv(
        flatten_runs(primary_with_response, seeds),
        output_dir / "main_expA_with_resp_runs.csv",
    )
    write_csv(
        flatten_runs(primary_without_response, seeds),
        output_dir / "main_expB_no_resp_runs.csv",
    )
    write_json(output_dir / "split_meta.json", split_metadata)

    statistical_comparisons: dict[str, Any] = {}
    for metric in ("f1", "ap"):
        with_response = np.asarray(
            [row["IF"][metric] for row in primary_with_response], dtype=float
        )
        without_response = np.asarray(
            [row["IF"][metric] for row in primary_without_response], dtype=float
        )
        paired_test = stats.ttest_rel(with_response, without_response)
        statistical_comparisons[
            f"IF_{metric.upper()}_with_vs_without_resp_h"
        ] = {
            "design": "Two-sided paired t-test across matched split/model seeds.",
            "metric": metric,
            "n_pairs": len(seeds),
            "seeds": list(seeds),
            f"mean_{metric}_with_resp_h": float(with_response.mean()),
            f"mean_{metric}_without_resp_h": float(without_response.mean()),
            "mean_paired_difference": float(
                (with_response - without_response).mean()
            ),
            "t_statistic": float(paired_test.statistic),
            "p_value_two_sided": float(paired_test.pvalue),
        }
    write_json(
        output_dir / "statistical_comparisons.json", statistical_comparisons
    )

    sensitivity_seeds = tuple(seeds[: min(N_SENSITIVITY_SEEDS, len(seeds))])
    timeout_tables = []
    for timeout_h in TIMEOUT_SWEEP:
        runs = []
        for seed in sensitivity_seeds:
            result, _ = one_run(
                cleaned,
                timeout_h,
                seed,
                include_resp=True,
                run_ocsvm=False,
            )
            runs.append(
                {
                    name: result[name]
                    for name in ("IF", "RF", "RuleCat80")
                    if name in result
                }
            )
        summary = aggregate(runs)
        summary.insert(0, "timeout_h", timeout_h)
        summary.insert(
            1, "prevalence", float((cleaned["resp_h"] > timeout_h).mean())
        )
        timeout_tables.append(summary)
    write_csv(pd.concat(timeout_tables, ignore_index=True), output_dir / "timeout_sweep.csv")

    prevalence_tables = []
    prevalence_run_tables = []
    composition_checks: list[dict[str, Any]] = []
    for target in PREVALENCE_TARGETS:
        runs = []
        metadata_rows = []
        for seed in sensitivity_seeds:
            result, metadata = one_run(
                cleaned,
                TIMEOUT_PRIMARY,
                seed,
                include_resp=True,
                prevalence_target=target,
                run_ocsvm=False,
            )
            runs.append({"IF": result["IF"], "RF": result["RF"]})
            metadata_rows.append(metadata)
            if target == 0.01 and metadata["sampling"] is not None:
                composition_checks.append(metadata["sampling"])
        summary = aggregate(runs)
        summary.insert(0, "prevalence_target", target)
        prevalence_tables.append(summary)
        prevalence_run_tables.append(
            flatten_runs(runs, sensitivity_seeds, prevalence_target=target)
        )
    write_csv(
        pd.concat(prevalence_tables, ignore_index=True),
        output_dir / "prevalence_sensitivity.csv",
    )
    write_csv(
        pd.concat(prevalence_run_tables, ignore_index=True),
        output_dir / "prevalence_sensitivity_runs.csv",
    )

    leakage_free_runs = []
    leakage_free_metadata = []
    for seed in sensitivity_seeds:
        result, metadata = one_run(
            cleaned,
            TIMEOUT_PRIMARY,
            seed,
            include_resp=False,
            prevalence_target=0.01,
            run_ocsvm=False,
        )
        leakage_free_runs.append({"IF": result["IF"]})
        leakage_free_metadata.append(metadata)
    leakage_free_summary = aggregate(leakage_free_runs)
    leakage_free_summary.insert(0, "prevalence_target", 0.01)
    write_csv(
        leakage_free_summary, output_dir / "prevalence_1pct_no_resp.csv"
    )
    write_csv(
        flatten_runs(
            leakage_free_runs, sensitivity_seeds, prevalence_target=0.01
        ),
        output_dir / "prevalence_1pct_no_resp_runs.csv",
    )

    fixed_seeds = tuple(seeds[: min(N_FIXED_CONTROL_SEEDS, len(seeds))])
    fixed_runs, fixed_summary = fixed_model_control(cleaned, fixed_seeds)
    write_csv(fixed_runs, output_dir / "fixed_model_control_runs.csv")
    write_csv(fixed_summary, output_dir / "fixed_model_control.csv")

    if run_benchmark:
        write_json(
            output_dir / "runtime_benchmark.json",
            runtime_benchmark(cleaned, seed=seeds[0]),
        )

    sampled_medians = [
        item["sampled_positive_distribution"]["median_h"]
        for item in composition_checks
    ]
    sampled_p90s = [
        item["sampled_positive_distribution"]["p90_h"]
        for item in composition_checks
    ]
    composition_payload = {
        "design": "Uniform random sampling without replacement; all negatives retained.",
        "per_seed": composition_checks,
        "reported_seed": composition_checks[0] if composition_checks else None,
        "across_seed_mean_sampled_median_h": float(np.mean(sampled_medians))
        if sampled_medians
        else None,
        "across_seed_mean_sampled_p90_h": float(np.mean(sampled_p90s))
        if sampled_p90s
        else None,
    }
    supplementary_checks = {
        "schema_version": "1.0",
        "data_sha256": audit["data_sha256"],
        "timeout_h": TIMEOUT_PRIMARY,
        "positive_sampling_composition": composition_payload,
        "fixed_model_control": {
            "design": (
                "Isolation Forest and validation-selected threshold are frozen at "
                "full prevalence; only test positives are downsampled."
            ),
            "seeds": list(fixed_seeds),
            "summary_file": "fixed_model_control.csv",
            "runs_file": "fixed_model_control_runs.csv",
            "summary": records_for_json(fixed_summary),
        },
        "leakage_free_1pct_retraining": {
            "design": (
                "Positive downsampling occurs before the split; Isolation Forest is "
                "retrained without resp_h and its threshold is reselected on validation."
            ),
            "seeds": list(sensitivity_seeds),
            "summary_file": "prevalence_1pct_no_resp.csv",
            "runs_file": "prevalence_1pct_no_resp_runs.csv",
            "summary": records_for_json(leakage_free_summary),
            "split_metadata": leakage_free_metadata,
        },
        "paired_statistical_comparisons": statistical_comparisons,
    }
    write_json(output_dir / "supplementary_checks.json", supplementary_checks)
    print(f"All Pipeline B outputs written to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.seeds) < 10:
        print(
            "WARNING: fewer than ten primary seeds were requested; outputs are not "
            "the canonical manuscript analysis.",
            file=sys.stderr,
        )
    run_protocol(
        args.data,
        args.output_dir,
        args.seeds,
        run_ocsvm=not args.skip_ocsvm,
        run_benchmark=not args.skip_runtime_benchmark,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
