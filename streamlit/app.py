#!/usr/bin/env python3
"""Simplified Streamlit companion to the canonical v2 experiments.

This application is intentionally separate from the manuscript analysis.  It uses a
single split for interactive exploration; canonical multi-seed results must be produced
with ``src/experiment_v2.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM


HERE = Path(__file__).resolve().parent
FEATURE_WHITELIST = ("resp_h", "hour", "weekday", "is_weekend")

st.set_page_config(
    page_title="Nurse-call-like anomaly-detection demo", layout="wide"
)
st.title("Runtime anomaly detection for nurse-call-like service logs")
st.warning(
    "Research demonstration only. This app is not a medical device and does not "
    "reproduce the manuscript's full multi-seed protocol."
)


@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    return pd.read_csv(HERE / "small_demo.csv", low_memory=False)


def locate_timestamp(columns: list[str], terms: tuple[str, ...]) -> str | None:
    lowered = {column.lower().strip(): column for column in columns}
    for key, original in lowered.items():
        if all(term in key for term in terms):
            return original
    return None


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    output = frame.copy()
    created_column = locate_timestamp(list(output.columns), ("created", "date"))
    closed_column = locate_timestamp(list(output.columns), ("closed", "date"))
    if "resp_h" not in output.columns:
        if created_column is None or closed_column is None:
            raise ValueError(
                "CSV must contain resp_h or created/closed date columns."
            )
        created = pd.to_datetime(output[created_column], errors="coerce")
        closed = pd.to_datetime(output[closed_column], errors="coerce")
        output["resp_h"] = (closed - created).dt.total_seconds() / 3600.0
    else:
        created = (
            pd.to_datetime(output[created_column], errors="coerce")
            if created_column
            else pd.Series(pd.NaT, index=output.index)
        )
    if "hour" not in output.columns:
        output["hour"] = created.dt.hour.fillna(0)
    if "weekday" not in output.columns:
        output["weekday"] = created.dt.weekday.fillna(0)
    output["is_weekend"] = (output["weekday"] >= 5).astype(int)

    audit = {
        "input_rows": len(output),
        "missing_closure_excluded": int(output["resp_h"].isna().sum()),
        "negative_intervals_excluded": int((output["resp_h"] < 0).sum()),
    }
    output = output[
        output["resp_h"].notna() & (output["resp_h"] >= 0)
    ].reset_index(drop=True)
    audit["retained_rows"] = len(output)
    return output, audit


def validation_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.percentile(scores, np.linspace(1, 99, 100)))
    values = [
        f1_score(labels, scores >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(values))])


st.sidebar.header("Data and label")
uploaded = st.sidebar.file_uploader("Upload a service-log CSV", type=("csv",))
raw = pd.read_csv(uploaded, low_memory=False) if uploaded is not None else load_default()
source_name = uploaded.name if uploaded is not None else "small_demo.csv"
try:
    data, audit = prepare(raw)
except ValueError as error:
    st.error(str(error))
    st.stop()

timeout_h = st.sidebar.number_input(
    "Delay-label timeout (hours)", min_value=0.1, max_value=168.0, value=2.0
)
if "is_anomaly" in data.columns:
    labels = data["is_anomaly"].astype(int).to_numpy()
    label_source = "uploaded is_anomaly column"
else:
    labels = (data["resp_h"] > timeout_h).astype(int).to_numpy()
    label_source = f"response time > {timeout_h:g} h"
if np.unique(labels).size != 2:
    st.error("The selected label contains only one class; adjust the data or timeout.")
    st.stop()

st.sidebar.header("Model")
model_name = st.sidebar.selectbox(
    "Detector",
    (
        "Isolation Forest (unsupervised)",
        "One-Class SVM (unsupervised)",
        "Random Forest (supervised reference)",
    ),
)
available_features = [name for name in FEATURE_WHITELIST if name in data.columns]
selected_features = st.sidebar.multiselect(
    "Features", available_features, default=available_features
)
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=101))
run = st.sidebar.button("Run demonstration", type="primary")

st.subheader("Data overview")
left, right = st.columns((1.4, 1.0))
with left:
    st.dataframe(data.head(10), use_container_width=True)
with right:
    st.json(
        {
            "source": source_name,
            "label": label_source,
            "prevalence": round(float(labels.mean()), 4),
            **audit,
        }
    )

figure, axis = plt.subplots(figsize=(6, 3))
axis.hist(data["resp_h"], bins=min(60, max(10, len(data) // 2)))
axis.set_xlabel("Response time (hours)")
axis.set_ylabel("Count")
axis.set_title("Response-time distribution after cleaning")
st.pyplot(figure, clear_figure=True)

if run:
    if not selected_features:
        st.error("Select at least one feature.")
        st.stop()
    features = data[selected_features].astype(float)
    indices = np.arange(len(data))
    train, remainder = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        stratify=labels[remainder],
    )

    if model_name.startswith("Isolation"):
        model = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed
        ).fit(features.iloc[train])
        validation_scores = -model.score_samples(features.iloc[validation])
        test_scores = -model.score_samples(features.iloc[test])
    elif model_name.startswith("One-Class"):
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(
            features.iloc[train]
        )
        validation_scores = -model.score_samples(features.iloc[validation])
        test_scores = -model.score_samples(features.iloc[test])
    else:
        model = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=1
        ).fit(features.iloc[train], labels[train])
        validation_scores = model.predict_proba(features.iloc[validation])[:, 1]
        test_scores = model.predict_proba(features.iloc[test])[:, 1]

    threshold = validation_threshold(validation_scores, labels[validation])
    predictions = (test_scores >= threshold).astype(int)
    st.subheader("Test-set results")
    metric_columns = st.columns(5)
    values = (
        ("Precision", precision_score(labels[test], predictions, zero_division=0)),
        ("Recall", recall_score(labels[test], predictions, zero_division=0)),
        ("F1", f1_score(labels[test], predictions, zero_division=0)),
        ("Average precision", average_precision_score(labels[test], test_scores)),
        ("Alert rate", predictions.mean()),
    )
    for column, (name, value) in zip(metric_columns, values):
        column.metric(name, f"{value:.3f}")

    plot_left, plot_right = st.columns(2)
    precision, recall, _ = precision_recall_curve(labels[test], test_scores)
    figure, axis = plt.subplots(figsize=(5, 3.5))
    axis.plot(recall, precision)
    axis.axhline(labels[test].mean(), linestyle="--", color="grey")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision–recall curve")
    plot_left.pyplot(figure, clear_figure=True)

    matrix = confusion_matrix(labels[test], predictions, labels=(0, 1))
    figure, axis = plt.subplots(figsize=(4, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks((0, 1), labels=("Normal", "Anomaly"))
    axis.set_yticks((0, 1), labels=("Normal", "Anomaly"))
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Confusion matrix")
    plot_right.pyplot(figure, clear_figure=True)

st.caption(
    "For canonical results, run src/experiment_v2.py. The demo is deliberately "
    "separate from the manuscript's multi-seed reproducibility protocol."
)
