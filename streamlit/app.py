# -*- coding: utf-8 -*-
# =============================================================================
# app.py — Streamlit interactive demo (v2)
#
# Part of the reproducibility package for:
#   "Simulation-Based Runtime Anomaly Detection for Nurse Call System Assurance:
#    A Reproducible Proof of Concept Using Public Service-Request Logs"
#
# Authors:
#   Yuanyuan (Maxine) Liu (1,2)  — corresponding author
#       jliu536@jh.edu | yl366@mynsu.nova.edu | ORCID 0000-0003-3410-6893
#   David R. Concepcion (1)
#   (1) Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA
#   (2) College of Computing and Engineering, Nova Southeastern University,
#       Fort Lauderdale, FL, USA
#
# Repository: https://github.com/maxineliu2020/ai-nursecall-runtime-anomaly-detection
# Archive:    https://doi.org/10.5281/zenodo.17767142 (all versions)
# License:    MIT
# =============================================================================
"""
Streamlit demo (v2) for:
Simulation-Based Runtime Anomaly Detection for Nurse Call System Assurance

Fixes relative to the previously deployed demo:
  1. Ground-truth evaluation only accepts a column literally named `is_anomaly`
     (previously auto-detected arbitrary 0/1 columns such as `is_weekend`).
  2. Feature whitelist: resp_h / hour / weekday / is_weekend only. Identifier-like
     columns (unique_key, bbl, ZIP codes, coordinates) are never used as features.
  3. Data cleaning matches the paper: negative response intervals and records with
     missing closure timestamps are excluded before labeling and modeling.
  4. Default labeling matches the paper: response time > timeout_h (default 2.0 h),
     with an optional quantile mode clearly marked as demo-only.

Usage: streamlit run streamlit/app.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FEATURE_WHITELIST = ["resp_h", "hour", "weekday", "is_weekend"]

st.set_page_config(page_title="Nurse-Call-like Anomaly Detection Demo", layout="wide")
st.title("📟 Runtime Anomaly Detection for Nurse-Call-like Service Logs (Demo v2)")
st.markdown(
    "Interactive companion to the manuscript. **Demo only** — simplified relative to the "
    "full experimental protocol (`src/experiment_v2.py`), which uses a train/validation/test "
    "split, validation-only threshold selection, and 10-seed confidence intervals."
)

# ---------------- data loading ----------------
@st.cache_data(show_spinner=False)
def load_default():
    for p in [ROOT / "DATA" / "erm2-nwe9.csv", ROOT / "DATA" / "small_demo.csv"]:
        if p.exists():
            return pd.read_csv(p, low_memory=False), p.name
    return None, None

st.sidebar.header("1. Data")
up = st.sidebar.file_uploader("Upload a service-log CSV (optional)", type=["csv"])
if up is not None:
    df_raw, src_name = pd.read_csv(up, low_memory=False), up.name
else:
    df_raw, src_name = load_default()
    if df_raw is None:
        st.error("No dataset found. Please upload a CSV.")
        st.stop()
st.sidebar.success(f"Loaded `{src_name}`: {len(df_raw):,} rows")

# ---------------- cleaning & features (paper-consistent) ----------------
def prepare(df):
    df = df.copy()
    audit = {}
    # locate timestamps (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}
    c_created = next((cols[k] for k in cols if "created" in k and "date" in k), None)
    c_closed = next((cols[k] for k in cols if "closed" in k and "date" in k), None)
    if "resp_h" not in df.columns:
        if c_created and c_closed:
            created = pd.to_datetime(df[c_created], errors="coerce")
            closed = pd.to_datetime(df[c_closed], errors="coerce")
            df["resp_h"] = (closed - created).dt.total_seconds() / 3600.0
            df["hour"] = created.dt.hour
            df["weekday"] = created.dt.weekday
        else:
            st.error("CSV must contain `resp_h`, or created/closed date columns.")
            st.stop()
    if "is_weekend" not in df.columns and "weekday" in df.columns:
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    n0 = len(df)
    audit["missing closure (excluded)"] = int(df["resp_h"].isna().sum())
    audit["negative intervals (excluded)"] = int((df["resp_h"] < 0).sum())
    df = df[df["resp_h"].notna() & (df["resp_h"] >= 0)].reset_index(drop=True)
    audit["retained"] = len(df)
    audit["input rows"] = n0
    return df, audit

df, audit = prepare(df_raw)
with st.expander("Data cleaning summary (same rules as the paper)", expanded=True):
    st.write(audit)

# ---------------- labeling ----------------
st.sidebar.header("2. Delay label")
label_mode = st.sidebar.radio("Label definition",
    ["Timeout threshold (paper default)", "Upper quantile (demo-only)"])
if label_mode.startswith("Timeout"):
    timeout_h = st.sidebar.number_input("timeout_h (hours)", 0.1, 168.0, 2.0, 0.1)
    df["is_anomaly_demo"] = (df["resp_h"] > timeout_h).astype(int)
    st.sidebar.caption("Note: in this dataset 2.0 h is close to the median response "
                       "time; see the paper's timeout-sensitivity analysis.")
else:
    q = st.sidebar.slider("Quantile", 0.80, 0.995, 0.95, 0.005)
    df["is_anomaly_demo"] = (df["resp_h"] > df["resp_h"].quantile(q)).astype(int)
st.sidebar.metric("Label prevalence", f"{df['is_anomaly_demo'].mean():.1%}")

# ---------------- model ----------------
st.sidebar.header("3. Model & settings")
model_name = st.sidebar.selectbox("Model",
    ["IsolationForest (unsupervised)", "OneClassSVM (unsupervised)",
     "RandomForest (supervised reference)"])
feat_opts = [c for c in FEATURE_WHITELIST if c in df.columns]
features = st.sidebar.multiselect("Features (identifier columns are never offered)",
                                  feat_opts, default=feat_opts)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.3, 0.05)
seed = int(st.sidebar.number_input("Random seed", 0, 9999, 42))
run = st.sidebar.button("Run experiment")

col_l, col_r = st.columns([1.15, 1.0])
with col_l:
    st.subheader("🧾 Data overview")
    st.dataframe(df.head(10))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["resp_h"], bins=60)
    ax.set_yscale("log")
    ax.set_xlabel("Response time (hours)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Response-time distribution (after cleaning)")
    st.pyplot(fig, clear_figure=True)

if run:
    if not features:
        st.error("Select at least one feature.")
        st.stop()
    X = df[features].astype(float).values
    y = df["is_anomaly_demo"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    if model_name.startswith("IsolationForest"):
        mdl = IsolationForest(n_estimators=200, contamination="auto",
                              random_state=seed).fit(Xtr)
        s_te = -mdl.score_samples(Xte)
    elif model_name.startswith("OneClassSVM"):
        cap = min(len(Xtr), 15000)
        sub = np.random.default_rng(seed).choice(len(Xtr), cap, replace=False)
        mdl = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr[sub])
        s_te = -mdl.score_samples(Xte)
    else:
        mdl = RandomForestClassifier(n_estimators=300, random_state=seed,
                                     n_jobs=-1).fit(Xtr, ytr)
        s_te = mdl.predict_proba(Xte)[:, 1]

    p, r, thr = precision_recall_curve(yte, s_te)
    ap = average_precision_score(yte, s_te)
    # demo operating point: max-F1 on the displayed split (retrospective; see paper
    # for validation-based selection)
    f1v = [f1_score(yte, (s_te >= t).astype(int)) for t in thr] if len(thr) else [0]
    best_t = float(thr[int(np.argmax(f1v))]) if len(thr) else 0.5
    yhat = (s_te >= best_t).astype(int)

    with col_r:
        st.subheader("📊 Results")
        st.markdown(f"**Model:** `{model_name}`  \n**AP:** `{ap:.3f}`  \n"
                    f"**Operating threshold:** `{best_t:.3f}` (max-F1, retrospective)")
        st.table(pd.DataFrame({
            "Precision": [precision_score(yte, yhat, zero_division=0)],
            "Recall": [recall_score(yte, yhat, zero_division=0)],
            "F1": [f1_score(yte, yhat, zero_division=0)],
            "Alert rate": [yhat.mean()],
        }).style.format("{:.3f}"))
        cm = confusion_matrix(yte, yhat)
        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        st.table(pd.DataFrame(cm,
                 index=["Actual normal", "Actual anomaly"],
                 columns=["Pred. normal", "Pred. anomaly"]))
        fig2, ax2 = plt.subplots(figsize=(5, 3.4))
        ax2.plot(r, p)
        ax2.axhline(yte.mean(), ls="--", c="grey", lw=1)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision–recall curve (AP = {ap:.3f})")
        st.pyplot(fig2, clear_figure=True)

    # ground truth: strict column-name match only
    if "is_anomaly" in df_raw.columns:
        st.subheader("4. Evaluation against provided ground truth (`is_anomaly`)")
        st.caption("Only a column literally named `is_anomaly` is treated as ground truth.")
    else:
        st.info("No `is_anomaly` column found in the uploaded file — ground-truth "
                "comparison skipped. (Labels above are constructed from the delay rule.)")
else:
    with col_r:
        st.info("Configure the sidebar and click **Run experiment**.")

st.caption("Demo only — for the full protocol (train/validation/test split, "
           "validation-only thresholds, 10-seed CIs, ablations, prevalence analysis) "
           "see `src/experiment_v2.py` in the repository.")
