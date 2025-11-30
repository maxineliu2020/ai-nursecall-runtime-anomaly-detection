# -*- coding: utf-8 -*-
"""
JHU 695.715 – Assured Autonomy — Course Project
Title: Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems
Author: Yuanyuan (Maxine) Liu
Instructor: David Concepcion
Term: Fall 2025

Streamlit demo app

This app provides an interactive UI for:
- Loading the provided NYC 311-style subset (erm2-nwe9.csv) or a user-uploaded CSV
- Computing response-time features (hours, weekday, hour-of-day)
- Training / evaluating anomaly detectors:
    * Isolation Forest (unsupervised)
    * One-Class SVM (unsupervised)
    * Random Forest (supervised baseline, requires labels)
- Visualising:
    * Dataset overview & distributions
    * Precision / Recall / F1 metrics
    * Precision–Recall curves
    * Threshold tuning for the Random Forest
The goal is to showcase the runtime anomaly-detection framework in a way that is
easy to explore and demo for reviewers / classmates / IEEE audience.
"""

import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Nurse-Call Runtime Anomaly Detection",
    page_icon="🩺",
    layout="wide",
)


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "DATA" / "erm2-nwe9.csv"


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATA)
    return df


def compute_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute response-time in hours + calendar features.
    Assumes columns:
      - 'Created Date'
      - 'Closed Date'
      - 'category'
    If an 'is_anomaly' column exists, it is preserved as ground truth.
    Otherwise we create a simple label based on a high quantile of resp_h.
    """
    df = df_raw.copy()

    # Robust datetime parsing
    df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
    df["Closed Date"] = pd.to_datetime(df["Closed Date"], errors="coerce")
    df = df.dropna(subset=["Created Date", "Closed Date"])

    # Response time (hours)
    df["resp_h"] = (df["Closed Date"] - df["Created Date"]).dt.total_seconds() / 3600.0

    # Basic calendar features
    df["hour"] = df["Created Date"].dt.hour
    df["weekday"] = df["Created Date"].dt.weekday  # 0 = Monday

    # If no label, create a simple anomaly label based on upper-quantile
    if "is_anomaly" not in df.columns:
        q = st.session_state.get("label_quantile", 0.9)
        thresh = df["resp_h"].quantile(q)
        df["is_anomaly"] = (df["resp_h"] >= thresh).astype(int)

    # Clean category
    if "category" not in df.columns:
        # Try to guess a category-like column
        for cand in ["Agency", "Department", "Category", "category_name"]:
            if cand in df.columns:
                df["category"] = df[cand].astype(str)
                break
        else:
            df["category"] = "unknown"

    df["category"] = df["category"].astype(str)

    return df


def build_feature_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Builds a sklearn Pipeline for feature preprocessing and returns:
        - preprocessor: fitted ColumnTransformer pipeline
        - X: transformed features
        - y: labels (is_anomaly)
    """
    feature_cols_num = ["resp_h", "hour", "weekday"]
    feature_cols_cat = ["category"]

    y = df["is_anomaly"].astype(int).values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )

    X = preprocessor.fit_transform(df[feature_cols_num + feature_cols_cat])

    return preprocessor, X, y


@st.cache_resource(show_spinner=False)
def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
    if_contamination: float = 0.1,
    ocsvm_nu: float = 0.1,
) -> Dict[str, object]:
    """
    Train three models:
        - Isolation Forest
        - One-Class SVM
        - Random Forest (supervised)
    Returns a dict of fitted models.
    """
    models: Dict[str, object] = {}

    # Isolation Forest
    if_model = IsolationForest(
        n_estimators=200,
        contamination=if_contamination,
        random_state=seed,
        n_jobs=-1,
    )
    if_model.fit(X_train)
    models["IF"] = if_model

    # One-Class SVM
    ocsvm_model = OneClassSVM(
        kernel="rbf",
        nu=ocsvm_nu,
        gamma="scale",
    )
    ocsvm_model.fit(X_train[y_train == 0])  # fit on normal only
    models["OCSVM"] = ocsvm_model

    # Random Forest (supervised baseline)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)
    models["RF"] = rf_model

    return models


def predict_scores(models: Dict[str, object], X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return anomaly scores / probabilities for each model.
    For IF / OCSVM: use decision_function (higher = more normal).
    For RF: use class probability for anomaly (class 1).
    We convert all to "anomaly score" where higher = more anomalous.
    """
    scores: Dict[str, np.ndarray] = {}

    if "IF" in models:
        s = -models["IF"].decision_function(X)  # higher = more anomalous
        scores["IF"] = s

    if "OCSVM" in models:
        s = -models["OCSVM"].decision_function(X)
        scores["OCSVM"] = s

    if "RF" in models:
        proba = models["RF"].predict_proba(X)[:, 1]  # P(y=1)
        scores["RF"] = proba

    return scores


def compute_pr_metrics(
    y_true: np.ndarray, score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)
    return precision, recall, ap


def compute_point_metrics(
    y_true: np.ndarray, score: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# --------------------------------------------------------------------------------------
# UI – Sidebar
# --------------------------------------------------------------------------------------

st.sidebar.title("⚙️ Experiment Controls")

# Data source
data_source = st.sidebar.radio(
    "Data source",
    ["Use bundled demo dataset", "Upload your own CSV"],
)

# Label quantile (for auto label when dataset has no is_anomaly)
st.sidebar.markdown("### Labeling (if `is_anomaly` is missing)")
label_q = st.sidebar.slider(
    "Upper quantile for anomaly label (resp_h)",
    0.80,
    0.99,
    0.90,
    0.01,
)
st.session_state["label_quantile"] = label_q

# Model hyperparameters
st.sidebar.markdown("### Model hyperparameters")

if_contamination = st.sidebar.slider(
    "Isolation Forest contamination",
    0.01,
    0.30,
    0.10,
    0.01,
)

ocsvm_nu = st.sidebar.slider(
    "OC-SVM ν (anomaly proportion)",
    0.01,
    0.30,
    0.10,
    0.01,
)

rf_threshold = st.sidebar.slider(
    "RF decision threshold (probability of anomaly)",
    0.10,
    0.90,
    0.50,
    0.01,
)

random_seed = st.sidebar.number_input("Random seed", 0, 9999, 0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "After adjusting the settings, click **Run / Re-run experiment** "
    "in the main panel to update all plots and metrics."
)

# --------------------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------------------

st.title("🩺 AI Nurse-Call Runtime Anomaly Detection & Assurance Demo")

st.markdown(
    """
This interactive app demonstrates the core ideas of the paper:

> **Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems**

You can explore the bundled demo dataset (NYC 311-style service tickets) or upload
your own nurse-call / service-call CSV with similar columns.
"""
)

# --- Data loading ---------------------------------------------------------------------

if data_source == "Use bundled demo dataset":
    if not DEFAULT_DATA.exists():
        st.error(f"Demo dataset not found at: `{DEFAULT_DATA}`")
        st.stop()
    df_raw = load_default_data()
    st.success(f"Loaded bundled dataset: `{DEFAULT_DATA.name}` "
               f"({len(df_raw):,} rows)")
else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Please upload a CSV to continue.")
        st.stop()
    df_raw = pd.read_csv(uploaded)
    st.success(f"Uploaded dataset with {len(df_raw):,} rows")

# --- Feature engineering --------------------------------------------------------------

try:
    df = compute_features(df_raw)
except Exception as e:
    st.exception(e)
    st.stop()

preprocessor, X_all, y_all = build_feature_pipeline(df)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.3,
    random_state=random_seed,
    stratify=y_all if len(np.unique(y_all)) > 1 else None,
)

# --- Dataset overview tab -------------------------------------------------------------

tab_overview, tab_metrics, tab_curves, tab_threshold = st.tabs(
    ["📊 Dataset overview", "📈 Model metrics", "📉 PR curves", "🎛 Threshold tuning"]
)

with tab_overview:
    st.subheader("Dataset snapshot")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.head(20))
    with c2:
        st.markdown("**Basic statistics**")
        st.write(
            pd.DataFrame(
                {
                    "n_rows": [len(df)],
                    "n_features": [X_all.shape[1]],
                    "anomaly_rate": [y_all.mean()],
                    "resp_h_mean": [df["resp_h"].mean()],
                    "resp_h_p90": [df["resp_h"].quantile(0.9)],
                }
            ).T.rename(columns={0: "value"})
        )

    st.markdown("### Response time distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["resp_h"], bins=50)
        ax.set_xlabel("Response time (hours)")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of response time")
        st.pyplot(fig)

    with col2:
        from scipy.stats import gaussian_kde

        xs = np.linspace(0, df["resp_h"].quantile(0.995), 200)
        kde = gaussian_kde(df["resp_h"])
        ys = kde(xs)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(xs, ys)
        ax2.set_xlabel("Response time (hours)")
        ax2.set_ylabel("Density")
        ax2.set_title("KDE of response time")
        st.pyplot(fig2)

# --- Run experiment button ------------------------------------------------------------

run = st.button("🚀 Run / Re-run experiment", type="primary")

if not run:
    st.stop()

with st.spinner("Training models and computing metrics..."):
    models = train_models(
        X_train,
        y_train,
        seed=random_seed,
        if_contamination=if_contamination,
        ocsvm_nu=ocsvm_nu,
    )
    scores_test = predict_scores(models, X_test)

# --------------------------------------------------------------------------------------
# Metrics tab
# --------------------------------------------------------------------------------------

with tab_metrics:
    st.subheader("Model comparison (on test set)")

    rows = []
    for name, score in scores_test.items():
        # choose an operating threshold:
        if name in ["IF", "OCSVM"]:
            # use median as a simple default
            th = np.median(score)
        else:  # RF
            th = rf_threshold
        m = compute_point_metrics(y_test, score, th)
        ap = average_precision_score(y_test, score)
        rows.append(
            {
                "Model": name,
                "Threshold": th,
                "Average Precision": ap,
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
            }
        )

    df_metrics = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_metrics.style.format("{:.3f}"))

    # nice metric cards for quick glance
    st.markdown("### Quick glance")

    cols = st.columns(len(scores_test))
    for col, (name, row) in zip(cols, df_metrics.iterrows()):
        col.metric(
            label=f"{name} AP",
            value=f"{row['Average Precision']:.3f}",
            delta=None,
        )

# --------------------------------------------------------------------------------------
# PR curves tab
# --------------------------------------------------------------------------------------

with tab_curves:
    st.subheader("Precision–Recall curves (test set)")

    fig, ax = plt.subplots(figsize=(7, 4))

    for name, score in scores_test.items():
        precision, recall, ap = compute_pr_metrics(y_test, score)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.markdown(
        """
Higher **Average Precision (AP)** and curves closer to the top-right corner
indicate better anomaly-detection performance.
"""
    )

# --------------------------------------------------------------------------------------
# Threshold tuning tab
# --------------------------------------------------------------------------------------

with tab_threshold:
    st.subheader("Random Forest threshold tuning (test set)")

    if "RF" not in scores_test:
        st.warning("Random Forest model not available.")
    else:
        rf_score = scores_test["RF"]

        # sweep many thresholds to plot precision / recall / f1
        ts = np.linspace(0.0, 1.0, 200)
        precs, recalls, f1s = [], [], []
        for t in ts:
            m = compute_point_metrics(y_test, rf_score, t)
            precs.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts, precs, label="Precision")
        ax.plot(ts, recalls, label="Recall")
        ax.plot(ts, f1s, label="F1")
        ax.axvline(rf_threshold, color="k", linestyle="--", alpha=0.6)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Random Forest threshold sweep")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # current operating point
        m_now = compute_point_metrics(y_test, rf_score, rf_threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{m_now['precision']:.3f}")
        c2.metric("Recall", f"{m_now['recall']:.3f}")
        c3.metric("F1", f"{m_now['f1']:.3f}")

        st.markdown(
            """
Use the **RF decision threshold** slider in the sidebar to explore different
operational trade-offs:

- Lower threshold → higher recall (catch more anomalies) but more false positives  
- Higher threshold → higher precision (fewer false alerts) but more missed anomalies
"""
        )

# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "JHU 695.715 Assured Autonomy – Course Project · "
    "Yuanyuan (Maxine) Liu · This demo is research code and not a medical device."
)
