
---

```python
# -*- coding: utf-8 -*-
"""
Streamlit demo for:
Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems

JHU 695.715 – Assured Autonomy — Course Project
Author: Yuanyuan (Maxine) Liu
Instructor: David Concepcion
Term: Fall 2025

Usage:
    streamlit run streamlit/app.py
"""

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """
    Load the cleaned CSV used in the paper (if available).
    Falls back gracefully if the file is missing.
    """
    default_path = ROOT / "results" / "real_calls_clean.csv"
    if default_path.exists():
        df = pd.read_csv(default_path)
        return df

    # Last-resort fallback: try the preprocessed CSV in DATA
    alt_path = ROOT / "DATA" / "erm2-nwe9.csv"
    if alt_path.exists():
        df = pd.read_csv(alt_path)
        return df

    raise FileNotFoundError(
        "No default dataset found. Please upload a CSV in the sidebar."
    )


def ensure_features(df: pd.DataFrame):
    """
    Ensure the minimal set of features exists.
    If some are missing, try to derive them or drop them gracefully.
    """
    # Resp_h: required
    if "resp_h" not in df.columns:
        # try to build from Created/Closed dates if available
        if {"Created Date", "Closed Date"}.issubset(df.columns):
            created = pd.to_datetime(df["Created Date"], errors="coerce")
            closed = pd.to_datetime(df["Closed Date"], errors="coerce")
            df["resp_h"] = (closed - created).dt.total_seconds() / 3600.0
        else:
            st.error(
                "The dataset must contain `resp_h` or both `Created Date` and `Closed Date`."
            )
            st.stop()

    # Temporal features (optional but useful)
    if "hour" not in df.columns and "Created Date" in df.columns:
        created = pd.to_datetime(df["Created Date"], errors="coerce")
        df["hour"] = created.dt.hour

    if "weekday" not in df.columns and "Created Date" in df.columns:
        created = pd.to_datetime(df["Created Date"], errors="coerce")
        df["weekday"] = created.dt.weekday

    # Category is optional
    if "category" not in df.columns and "Agency" in df.columns:
        df["category"] = df["Agency"]

    # Build a simple label if none is provided (unsupervised baseline)
    if "is_anomaly" not in df.columns:
        q = df["resp_h"].quantile(0.95)
        df["is_anomaly"] = (df["resp_h"] > q).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Construct X, y for modeling.
    """
    feature_cols = []
    for col in ["resp_h", "hour", "weekday"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].values.astype(float)
    y = df["is_anomaly"].values.astype(int)
    return X, y, feature_cols


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="Nurse-Call Anomaly Detection Demo",
    layout="wide",
)

st.title("📟 Runtime Anomaly Detection for Nurse-Call–like Service Tickets")

st.markdown(
    """
This Streamlit app provides an **interactive front-end** for the course project:

> *Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems*  
> JHU 695.715 – Assured Autonomy (Fall 2025)
"""
)

# Sidebar: data selection -----------------------------------------------------
st.sidebar.header("1. Data")

use_default = st.sidebar.checkbox(
    "Use built-in NYC 311 subset (recommended to start)", value=True
)

uploaded_file = st.sidebar.file_uploader(
    "Or upload your own preprocessed CSV", type=["csv"]
)

if use_default:
    try:
        df_raw = load_default_data()
    except FileNotFoundError as e:
        st.warning(str(e))
        use_default = False
        df_raw = None
else:
    df_raw = None

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    use_default = False

if df_raw is None:
    st.stop()

df = ensure_features(df_raw.copy())
X, y, feature_cols = get_feature_matrix(df)

st.sidebar.success(f"Loaded dataset with {len(df):,} rows.")


# Sidebar: model selection ----------------------------------------------------
st.sidebar.header("2. Model & Settings")

model_name = st.sidebar.selectbox(
    "Model",
    ["IsolationForest (unsupervised)", "OneClassSVM (unsupervised)", "RandomForest (supervised)"],
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)

run_button = st.sidebar.button("Run experiment")


# Main layout: 2 columns ------------------------------------------------------
col_left, col_right = st.columns([1.2, 1.0])

with col_left:
    st.subheader("🧾 Data overview")
    st.write(df.head())
    st.write("Feature columns used:", feature_cols)

    # Basic histogram of response time
    fig, ax = plt.subplots()
    ax.hist(df["resp_h"], bins=60)
    ax.set_xlabel("Response time (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of response time")
    st.pyplot(fig, clear_figure=True)

    # Category distribution (if available)
    if "category" in df.columns:
        st.markdown("**Top categories**")
        cat_counts = df["category"].value_counts().head(10)
        st.bar_chart(cat_counts)


# Run experiment --------------------------------------------------------------
if run_button:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_name.startswith("IsolationForest"):
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=random_state,
        )
        model.fit(X_train)
        scores_test = -model.decision_function(X_test)

    elif model_name.startswith("OneClassSVM"):
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        model.fit(X_train)
        scores_test = -model.decision_function(X_test)

    else:  # RandomForest
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        # Use probability of class 1 as anomaly score
        scores_test = model.predict_proba(X_test)[:, 1]

    # Precision–recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, scores_test)
    ap = average_precision_score(y_test, scores_test)

    # Choose a default operating point: max F1
    f1_values = []
    for t in thresholds:
        y_pred = (scores_test >= t).astype(int)
        f1_values.append(f1_score(y_test, y_pred))
    if len(thresholds) > 0:
        best_idx = int(np.argmax(f1_values))
        best_th = float(thresholds[best_idx])
    else:
        best_th = 0.5

    y_pred_best = (scores_test >= best_th).astype(int)

    prec = precision_score(y_test, y_pred_best)
    rec = recall_score(y_test, y_pred_best)
    f1 = f1_score(y_test, y_pred_best)

    with col_right:
        st.subheader("📊 Model results")

        st.markdown(
            f"""
**Model:** `{model_name}`  
**Average Precision (AP):** `{ap:.3f}`  
**Chosen threshold:** `{best_th:.3f}` (max F1)
"""
        )

        st.table(
            pd.DataFrame(
                {
                    "Precision": [prec],
                    "Recall": [rec],
                    "F1": [f1],
                }
            ).style.format("{:.3f}")
        )

        # Plot PR curve
        fig2, ax2 = plt.subplots()
        ax2.plot(recall, precision)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision–Recall curve ({model_name})")
        st.pyplot(fig2, clear_figure=True)

        # Show anomaly rate
        st.markdown(
            f"Anomaly rate in test set: **{y_test.mean():.3%}**  "
            f"(labels from `is_anomaly`)."
        )

else:
    with col_right:
        st.info("Configure the model in the sidebar and click **Run experiment**.")
