#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline A: corrected synthetic anomaly-injection experiment.

The implementation preserves the archived injection design while using independent
seeded datasets, stratified 60/20/20 splits, validation-only threshold selection,
and transparent treatment of removed (therefore row-unobservable) events.

Outputs are overwritten deterministically on each run; rerunning the script never
appends duplicate summary rows.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

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


PRIMARY_SEEDS = tuple(range(201, 211))
INJECTION_RATES = (0.05, 0.10, 0.20)
N_EVENTS = 1_000


def repository_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() in {"src", "scripts"} else here


REPOSITORY_ROOT = repository_root()
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "pipeline_a"


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Provide a non-empty list of unique integers.")
    return values


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(not 0 < item < 1 for item in values):
        raise argparse.ArgumentTypeError("Rates must be unique values between 0 and 1.")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Rates must be unique.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the corrected Pipeline A synthetic-injection protocol."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory."
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=PRIMARY_SEEDS,
        help="Comma-separated seeds (default: 201,...,210).",
    )
    parser.add_argument(
        "--rates",
        type=parse_float_list,
        default=INJECTION_RATES,
        help="Comma-separated nominal injection rates (default: .05,.10,.20).",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=N_EVENTS,
        help="Base events per seeded dataset (canonical value: 1000).",
    )
    parser.add_argument(
        "--skip-runtime-benchmark", action="store_true", help="Skip latency output."
    )
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
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


def generate(
    seed: int,
    *,
    n_events: int = N_EVENTS,
    rooms: int = 50,
    emergency_ratio: float = 0.20,
) -> tuple[pd.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 6, 8, 0, 0)
    rows = []
    for index in range(n_events):
        emergency = rng.random() < emergency_ratio
        delay = max(1.0, rng.normal(25 if emergency else 45, 10))
        rows.append(
            {
                "timestamp": start + timedelta(seconds=index * 5),
                "room_id": 100 + int(rng.integers(rooms)),
                "call_type": "emergency" if emergency else "normal",
                "response_delay_s": float(delay),
                "label": 0,
                "anomaly_type": "normal",
            }
        )
    return pd.DataFrame.from_records(rows), rng


def inject(
    frame: pd.DataFrame, rate: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Inject the five archived anomaly types and report injected counts."""
    output = frame.copy()
    n_selected = max(1, int(len(output) * rate))
    selected = rng.choice(output.index.to_numpy(), size=n_selected, replace=False)
    side_count = max(1, n_selected // 10)
    removed = set(rng.choice(selected, size=side_count, replace=False))
    remaining = np.asarray(sorted(set(selected) - removed))
    shifted = set(
        rng.choice(remaining, size=min(side_count, len(remaining)), replace=False)
    )
    other = sorted(set(selected) - removed - shifted)

    added_rows: list[pd.Series] = []
    injected_counts = {
        "delayed_response": 0,
        "repeated_call_rows": 0,
        "volume_burst_rows": 0,
        "invalid_timestamp": len(shifted),
        "missing_response_unobservable": len(removed),
    }
    for index in other:
        anomaly_type = ("delay", "repeat", "volume")[int(rng.integers(3))]
        if anomaly_type == "delay":
            output.at[index, "response_delay_s"] = float(rng.uniform(100, 240))
            output.at[index, "label"] = 1
            output.at[index, "anomaly_type"] = "delay"
            injected_counts["delayed_response"] += 1
        elif anomaly_type == "repeat":
            base = output.loc[index].copy()
            duplicate_count = int(rng.integers(2, 4))
            for duplicate in range(duplicate_count):
                row = base.copy()
                row["timestamp"] = base["timestamp"] + timedelta(seconds=10 * duplicate)
                row["label"] = 1
                row["anomaly_type"] = "repeat"
                added_rows.append(row)
            injected_counts["repeated_call_rows"] += duplicate_count
        else:
            base = output.loc[index].copy()
            for _ in range(10):
                row = base.copy()
                row["timestamp"] = base["timestamp"] + timedelta(
                    seconds=int(rng.integers(0, 60))
                )
                row["label"] = 1
                row["anomaly_type"] = "volume"
                added_rows.append(row)
            injected_counts["volume_burst_rows"] += 10

    output = output.drop(index=list(removed))
    for index in shifted:
        output.at[index, "timestamp"] = output.at[index, "timestamp"] - timedelta(
            minutes=int(rng.integers(5, 30))
        )
        output.at[index, "label"] = 1
        output.at[index, "anomaly_type"] = "timestamp"
    if added_rows:
        output = pd.concat((output, pd.DataFrame(added_rows)), ignore_index=True)
    output = output.sort_values("timestamp").reset_index(drop=True)
    return output, injected_counts


def featurize(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["delay"] = frame["response_delay_s"].astype(float)
    features["is_emergency"] = (frame["call_type"] == "emergency").astype(int)
    timestamps = pd.to_datetime(frame["timestamp"])
    features["minute"] = timestamps.dt.minute
    features["second"] = timestamps.dt.second
    features["room_mod_10"] = frame["room_id"].astype(int) % 10
    return features


def pick_threshold_val(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.percentile(scores, np.linspace(1, 99.5, 200)))
    f1_values = [
        f1_score(labels, scores >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(f1_values))])


def metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    anomaly_types: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "alert_rate": float(np.mean(predictions)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if scores is not None and np.unique(labels).size > 1:
        result["ap"] = average_precision_score(labels, scores)
        result["auc_roc"] = roc_auc_score(labels, scores)
    for anomaly_type in ("delay", "repeat", "volume", "timestamp"):
        selected = anomaly_types == anomaly_type
        if selected.sum() > 0:
            result[f"recall_{anomaly_type}"] = float(predictions[selected].mean())
            result[f"n_test_{anomaly_type}"] = int(selected.sum())
    return result


def mean_ci(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size > 1:
        half_width = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
        standard_deviation = float(array.std(ddof=1))
    else:
        half_width = float("nan")
        standard_deviation = 0.0
    return mean, mean - half_width, mean + half_width, standard_deviation


def aggregate(rows: Sequence[dict[str, dict[str, Any]]]) -> pd.DataFrame:
    records = []
    models = sorted({model for row in rows for model in row})
    for model in models:
        model_rows = [row[model] for row in rows if model in row]
        record: dict[str, Any] = {"model": model, "n_runs": len(model_rows)}
        names = sorted({name for row in model_rows for name in row})
        for name in names:
            values = [float(row[name]) for row in model_rows if name in row]
            mean, low, high, standard_deviation = mean_ci(values)
            record[f"{name}_mean"] = mean
            record[f"{name}_sd"] = standard_deviation
            record[f"{name}_ci_lo"] = low
            record[f"{name}_ci_hi"] = high
        records.append(record)
    return pd.DataFrame.from_records(records)


def one_run(
    rate: float, seed: int, *, n_events: int = N_EVENTS
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base, rng = generate(seed, n_events=n_events)
    frame, injection_counts = inject(base, rate, rng)
    features = featurize(frame)
    labels = frame["label"].to_numpy()
    anomaly_types = frame["anomaly_type"].to_numpy()
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

    result: dict[str, dict[str, Any]] = {}
    isolation_forest = IsolationForest(
        n_estimators=200, contamination="auto", random_state=seed
    ).fit(features.iloc[train])
    validation_scores = -isolation_forest.score_samples(features.iloc[validation])
    test_scores = -isolation_forest.score_samples(features.iloc[test])
    threshold = pick_threshold_val(validation_scores, labels[validation])
    result["IF"] = metrics(
        labels[test],
        (test_scores >= threshold).astype(int),
        test_scores,
        anomaly_types[test],
    )

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10).fit(
        features.iloc[train]
    )
    validation_scores = -ocsvm.score_samples(features.iloc[validation])
    test_scores = -ocsvm.score_samples(features.iloc[test])
    threshold = pick_threshold_val(validation_scores, labels[validation])
    result["OCSVM"] = metrics(
        labels[test],
        (test_scores >= threshold).astype(int),
        test_scores,
        anomaly_types[test],
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    ).fit(features.iloc[train], labels[train])
    validation_scores = random_forest.predict_proba(features.iloc[validation])[:, 1]
    test_scores = random_forest.predict_proba(features.iloc[test])[:, 1]
    threshold = pick_threshold_val(validation_scores, labels[validation])
    result["RF"] = metrics(
        labels[test],
        (test_scores >= threshold).astype(int),
        test_scores,
        anomaly_types[test],
    )

    rule_predictions = (frame["response_delay_s"].to_numpy()[test] > 60).astype(int)
    result["Rule60s"] = metrics(
        labels[test], rule_predictions, None, anomaly_types[test]
    )

    metadata = {
        "seed": seed,
        "nominal_injection_rate": rate,
        "base_events": n_events,
        "n_after_injection": len(frame),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "effective_prevalence": float(labels.mean()),
        "injection_counts": injection_counts,
    }
    return result, metadata


def flatten_runs(
    rows: Sequence[dict[str, dict[str, Any]]],
    seeds: Sequence[int],
    rate: float,
) -> pd.DataFrame:
    records = []
    for seed, result in zip(seeds, rows):
        for model, model_metrics in result.items():
            records.append(
                {
                    "nominal_injection_rate": rate,
                    "seed": seed,
                    "model": model,
                    **model_metrics,
                }
            )
    return pd.DataFrame.from_records(records)


def latency_benchmark(
    *, seed: int = 201, n_events: int = N_EVENTS, repetitions: int = 200
) -> dict[str, Any]:
    base, rng = generate(seed, n_events=n_events)
    frame, _ = inject(base, 0.10, rng)
    features = featurize(frame)
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=seed
    ).fit(features)
    one_record = features.iloc[[0]]
    latencies = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model.score_samples(one_record)
        latencies.append(time.perf_counter() - start)
    latency_ms = np.asarray(latencies) * 1000.0
    start = time.perf_counter()
    model.score_samples(features)
    batch_seconds = time.perf_counter() - start
    return {
        "single_ms_mean": float(latency_ms.mean()),
        "single_ms_p95": float(np.percentile(latency_ms, 95)),
        "batch_events_per_s": len(features) / batch_seconds,
        "model_size_mb": len(pickle.dumps(model)) / 1e6,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
    }


def run_protocol(
    output_dir: Path,
    seeds: Sequence[int],
    rates: Sequence[float],
    *,
    n_events: int,
    run_benchmark: bool,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    raw_tables = []
    metadata_rows = []
    for rate in rates:
        runs = []
        for seed in seeds:
            result, metadata = one_run(rate, seed, n_events=n_events)
            runs.append(result)
            metadata_rows.append(metadata)
        summary = aggregate(runs)
        summary.insert(0, "nominal_injection_rate", rate)
        summaries.append(summary)
        raw_tables.append(flatten_runs(runs, seeds, rate))
        effective = [
            item["effective_prevalence"]
            for item in metadata_rows
            if item["nominal_injection_rate"] == rate
        ]
        print(
            f"rate={rate:.2f}: mean effective prevalence={np.mean(effective):.3f}"
        )

    write_csv(pd.concat(summaries, ignore_index=True), output_dir / "synth_metrics.csv")
    write_csv(
        pd.concat(raw_tables, ignore_index=True), output_dir / "synth_metrics_runs.csv"
    )
    write_json(output_dir / "run_meta.json", metadata_rows)
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
            "scipy": __import__("scipy").__version__,
        },
    )
    if run_benchmark:
        write_json(
            output_dir / "latency.json",
            latency_benchmark(seed=seeds[0], n_events=n_events),
        )
    print(f"All Pipeline A outputs written to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.seeds) < 10 or args.events != N_EVENTS:
        print(
            "WARNING: non-canonical seed count or event count requested.", file=sys.stderr
        )
    run_protocol(
        args.output_dir,
        args.seeds,
        args.rates,
        n_events=args.events,
        run_benchmark=not args.skip_runtime_benchmark,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
