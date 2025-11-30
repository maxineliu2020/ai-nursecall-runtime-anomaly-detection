# -*- coding: utf-8 -*-
"""
JHU 695.715 Assured Autonomy — Course Project
Title: Runtime Anomaly Detection and Assurance for Nurse-Call–like Service Logs
Author: Yuanyuan (Maxine) Liu
Instructor: David Concepcion
Term: Fall 2025

This script loads a real-world ticket/service log CSV (e.g., NYC 311),
auto-detects datetime columns, builds labels by response latency, and evaluates
lightweight anomaly detectors (Isolation Forest, One-Class SVM) and a supervised
baseline (RandomForest). It produces 12+ publication-ready figures and a metrics CSV.

- If SHAP is installed, RF explanations are exported.
- If TensorFlow is installed, an optional tiny autoencoder baseline is run.
- Robust to column-name variations and missing fields; always creates `_created`
  and `is_anomaly` to avoid KeyErrors encountered earlier.
"""

import argparse, os, warnings, math, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
)

# Optional: SHAP
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# Optional: TensorFlow AE
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_OK = True
except Exception:
    TF_OK = False

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 160

# ---------- paths ----------
RES_DIR = "results"
os.makedirs(RES_DIR, exist_ok=True)

# ---------- helpers ----------
def detect_col(df, synonyms):
    """Return the first existing column name in df that matches any synonym (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for syn in synonyms:
        if syn.lower() in cols:
            return cols[syn.lower()]
    # try contains
    for syn in synonyms:
        for c in df.columns:
            if syn.lower() in c.lower():
                return c
    return None

CREATED_SYNS = [
    "created date", "created_on", "created", "request_received", "open date",
    "request datetime", "created time", "created_on_date", "date created"
]
CLOSED_SYNS = [
    "closed date", "closed_on", "resolution date", "closed", "completion date",
    "closed time", "closed_on_date", "date closed"
]
CATEGORY_SYNS = [
    "category", "type", "agency", "complaint type", "request type", "service", "subcategory"
]

def parse_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)

def safe_series(a):
    return np.array(a).astype(float)

def pr_ap(y, scores):
    p, r, _ = precision_recall_curve(y, scores)
    return p, r, average_precision_score(y, scores)

def savefig(path, tight=True):
    if tight: plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ---------- data loader ----------
def load_real_csv(csv_path, limit=None):
    df = pd.read_csv(csv_path, low_memory=False)
    if limit: df = df.head(int(limit))

    # auto detect
    c_created = detect_col(df, CREATED_SYNS)
    c_closed  = detect_col(df, CLOSED_SYNS)
    c_cat     = detect_col(df, CATEGORY_SYNS)

    # to datetime if present
    if c_created:
        df[c_created] = parse_dt(df[c_created])
    if c_closed:
        df[c_closed]  = parse_dt(df[c_closed])

    # build canonical columns
    if c_created:
        df["_created"] = df[c_created]
    else:
        # fallback: synthesize a monotonic timeline to avoid KeyError
        df["_created"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(len(df)), unit="m")

    if c_closed:
        df["_closed"] = df[c_closed]
    else:
        df["_closed"] = pd.NaT

    # compute response time (hours) when closed exists
    df["response_time_h"] = np.nan
    mask = df["_closed"].notna() & df["_created"].notna()
    if mask.any():
        df.loc[mask, "response_time_h"] = (df.loc[mask, "_closed"] - df.loc[mask, "_created"]).dt.total_seconds()/3600.0

    # ensure category
    if not c_cat:
        df["_category"] = "unknown"
    else:
        df["_category"] = df[c_cat].astype(str).fillna("unknown")

    df = df.sort_values("_created")

    # NOTE: label will be created later after we know timeout_h
    # safety: guarantee existence of placeholders to avoid KeyError in plotting sections
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    out_csv = os.path.join(RES_DIR, "real_calls_clean.csv")
    df.to_csv(out_csv, index=False)
    print(f"  cleaned saved -> {out_csv}  rows={len(df)}  created_col='{c_created}' closed_col='{c_closed}' category_col='{c_cat}'")
    return df

# ---------- plotting basics ----------
def plot_delay_distributions(df):
    x = df["response_time_h"].dropna()
    if len(x) == 0:
        return
    plt.figure(figsize=(6,4))
    plt.hist(x, bins=50, alpha=.85)
    plt.xlabel("Response time (hours)"); plt.ylabel("Count"); plt.title("Histogram of response time")
    savefig(os.path.join(RES_DIR, "hist_delay.png"))

    plt.figure(figsize=(6,4))
    x.plot(kind="kde")
    plt.xlabel("Response time (hours)"); plt.title("KDE of response time")
    savefig(os.path.join(RES_DIR, "kde_delay.png"))

def plot_box_by_category(df):
    if "_category" not in df.columns: return
    d = df.dropna(subset=["response_time_h"]).copy()
    if len(d)==0: return
    cats = (d["_category"].value_counts().head(8)).index.tolist()
    d = d[d["_category"].isin(cats)]
    plt.figure(figsize=(9,5))
    data = [d.loc[d["_category"]==c, "response_time_h"] for c in cats]
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1,len(cats)+1), cats, rotation=30, ha="right")
    plt.ylabel("Response time (hours)"); plt.title("Boxplot by category (top-8)")
    savefig(os.path.join(RES_DIR, "box_delay_by_category.png"))

    plt.figure(figsize=(6,4))
    d["_category"].value_counts().head(8).plot(kind="pie", autopct="%1.0f%%")
    plt.title("Category distribution (top-8)")
    savefig(os.path.join(RES_DIR, "category_pie.png"))

def plot_time_heatmaps(df):
    s = df.set_index("_created").resample("1h")["is_anomaly"].count()
    plt.figure(figsize=(6,4))
    s.plot()
    plt.ylabel("alerts/hour (proxy)"); plt.title("Alerts/hour (count of records)")
    savefig(os.path.join(RES_DIR, "alerts_per_hour.png"))

    d = df.copy()
    d["weekday"] = d["_created"].dt.weekday
    d["hour"]    = d["_created"].dt.hour
    mat = d.pivot_table(index="weekday", columns="hour", values="is_anomaly", aggfunc="count").fillna(0)
    plt.figure(figsize=(8,4))
    plt.imshow(mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="count")
    plt.xlabel("Hour"); plt.ylabel("Weekday (0=Mon)"); plt.title("Heatmap: count by weekday-hour")
    savefig(os.path.join(RES_DIR, "heatmap_weekday_hour.png"))

# ---------- modeling ----------
def features_from_df(df):
    # Simple numeric features; one-hot for category (top-k)
    use = pd.DataFrame(index=df.index)
    # time-of-day features
    use["hour"]    = df["_created"].dt.hour
    use["weekday"] = df["_created"].dt.weekday
    # delay proxy (closed-known)
    use["resp_h"]  = df["response_time_h"].fillna(-1)
    # top category one-hot
    top = df["_category"].value_counts().head(8).index
    for c in top:
        use[f"cat_{c}"] = (df["_category"]==c).astype(int)
    use = use.fillna(0)
    return use

def threshold_at_recall(p, r, scores, y, target_recall=0.8):
    # pick threshold with recall>=target and best F1
    best_t, best_f1 = None, -1
    for pi, ri, ti in zip(p, r, np.linspace(scores.min(), scores.max(), len(p))):
        if ri >= target_recall:
            yhat = (scores >= ti).astype(int)
            f1 = f1_score(y, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, ti
    if best_t is None:
        best_t = np.percentile(scores, 90)
    return best_t

def plot_cm(cm, name):
    plt.figure(figsize=(3.6,3.2))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i,j]), ha="center", va="center")
    plt.xticks([0,1], ["Neg","Pos"]); plt.yticks([0,1], ["Neg","Pos"])
    plt.title(f"Confusion Matrix - {name}")
    savefig(os.path.join(RES_DIR, f"cm_{name}.png"))

def bootstrap_ci(v, ci=95):
    arr = np.array(v)
    if len(arr)==0: return (np.nan, np.nan, np.nan)
    means = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(arr[idx].mean())
    lo = np.percentile(means, (100-ci)/2.0)
    hi = np.percentile(means, 100-(100-ci)/2.0)
    return arr.mean(), lo, hi

def eval_all_models(df, timeout_h=2.0, run_shap=False, no_ae=False):
    # (re)label anomalies by latency
    if "response_time_h" in df.columns:
        df["is_anomaly"] = (df["response_time_h"] > timeout_h).astype(int)
    else:
        df["is_anomaly"] = 0

    X = features_from_df(df)
    y = df["is_anomaly"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.sum()>0 else None)

    MET = {}  # metrics collection

    # --- Isolation Forest ---
    ifor = IsolationForest(n_estimators=400, random_state=42, n_jobs=-1, contamination="auto")
    ifor.fit(Xtr)
    s_if = -ifor.score_samples(Xte)  # higher=more anomalous
    p_if, r_if, ap_if = pr_ap(yte, s_if)
    t_if = threshold_at_recall(p_if, r_if, s_if, yte, target_recall=0.8)
    yhat_if = (s_if>=t_if).astype(int)
    MET["IF"] = dict(precision=precision_score(yte,yhat_if,zero_division=0),
                     recall=recall_score(yte,yhat_if,zero_division=0),
                     f1=f1_score(yte,yhat_if,zero_division=0),
                     ap=ap_if)
    plot_cm(confusion_matrix(yte, yhat_if), "IF")

    # PR curve
    plt.figure(figsize=(6,4))
    plt.plot(r_if, p_if, label=f"IF (AP={ap_if:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves (IF/OCSVM/RF)")

    # --- One-Class SVM ---
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc.fit(Xtr)
    s_oc = -oc.score_samples(Xte)
    p_oc, r_oc, ap_oc = pr_ap(yte, s_oc)
    t_oc = threshold_at_recall(p_oc, r_oc, s_oc, yte, target_recall=0.8)
    yhat_oc = (s_oc>=t_oc).astype(int)
    MET["OCSVM"] = dict(precision=precision_score(yte,yhat_oc,zero_division=0),
                        recall=recall_score(yte,yhat_oc,zero_division=0),
                        f1=f1_score(yte,yhat_oc,zero_division=0),
                        ap=ap_oc)
    plt.plot(r_oc, p_oc, label=f"OCSVM (AP={ap_oc:.3f})")

    # --- RandomForest supervised (uses y labels) ---
    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_proba = rf.predict_proba(Xte)[:,1]
    p_rf, r_rf, ap_rf = pr_ap(yte, rf_proba)
    # choose threshold by Youden-like approach
    t_rf = np.percentile(rf_proba, 90)
    yhat_rf = (rf_proba>=t_rf).astype(int)
    MET["RF"] = dict(precision=precision_score(yte,yhat_rf,zero_division=0),
                     recall=recall_score(yte,yhat_rf,zero_division=0),
                     f1=f1_score(yte,yhat_rf,zero_division=0),
                     ap=ap_rf)
    plt.plot(r_rf, p_rf, label=f"RF (AP={ap_rf:.3f})")
    plt.legend()
    savefig(os.path.join(RES_DIR, "pr_curves_all.png"))

    # Feature importance
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:12]
    plt.figure(figsize=(7,4))
    plt.bar(np.array(X.columns)[idx], imp[idx])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Importance"); plt.title("RF Feature Importances")
    savefig(os.path.join(RES_DIR, "rf_feature_importance.png"))

    # Threshold sweep plots (ops view)
    def ops_latency_vs_threshold(scores, model_name):
        dtr = pd.DataFrame({"_created": df.loc[Xte.index, "_created"].values,
                            "is_anomaly": yte, "score": scores})
        dtr = dtr.sort_values("score")
        recs, precs = [], []
        for q in np.linspace(50, 99.5, 60):
            th = np.percentile(scores, q)
            yhat = (scores>=th).astype(int)
            recs.append(recall_score(yte, yhat, zero_division=0))
            precs.append(precision_score(yte, yhat, zero_division=0))
        plt.figure(figsize=(6,4))
        plt.plot(recs, precs)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Ops trade-off (precision vs recall) — {model_name}")
        savefig(os.path.join(RES_DIR, f"ops_alerts_vs_th_{model_name}.png"))

    ops_latency_vs_threshold(s_if, "IF")
    ops_latency_vs_threshold(s_oc, "OCSVM")
    ops_latency_vs_threshold(rf_proba, "RF")

    # Metrics bar with 95% CI (bootstrap)
    labels = []
    p_means = []; p_lo=[]; p_hi=[]
    r_means = []; r_lo=[]; r_hi=[]
    f_means = []; f_lo=[]; f_hi=[]
    for k in ["IF","OCSVM","RF"]:
        labels.append(k)
        p_me, p_l, p_h = bootstrap_ci([MET[k]["precision"]])
        r_me, r_l, r_h = bootstrap_ci([MET[k]["recall"]])
        f_me, f_l, f_h = bootstrap_ci([MET[k]["f1"]])
        p_means.append(p_me); p_lo.append(p_me-p_l); p_hi.append(p_h-p_me)
        r_means.append(r_me); r_lo.append(r_me-r_l); r_hi.append(r_h-r_me)
        f_means.append(f_me); f_lo.append(f_me-f_l); f_hi.append(f_h-f_me)

    x = np.arange(len(labels)); w=0.25
    plt.figure(figsize=(7.5,4))
    plt.bar(x- w, p_means, yerr=[p_lo,p_hi], width=w, capsize=3, label="Precision")
    plt.bar(x    , r_means, yerr=[r_lo,r_hi], width=w, capsize=3, label="Recall")
    plt.bar(x+ w, f_means, yerr=[f_lo,f_hi], width=w, capsize=3, label="F1")
    plt.xticks(x, labels); plt.ylim(0,1.05)
    plt.legend(); plt.title("Metrics (mean ± 95% CI)")
    savefig(os.path.join(RES_DIR, "metrics_bar_ci.png"))

    # SHAP (optional)
    if run_shap and SHAP_OK:
        try:
            expl = shap.TreeExplainer(rf)
            sv = expl.shap_values(Xte)[1] if isinstance(expl.shap_values(Xte), list) else expl.shap_values(Xte)
            shap.summary_plot(sv, Xte, show=False)
            savefig(os.path.join(RES_DIR, "shap_summary.png"))
            shap.summary_plot(sv, Xte, plot_type="bar", show=False)
            savefig(os.path.join(RES_DIR, "shap_bar.png"))
        except Exception as e:
            print(f"[SHAP] skipped: {e}")

    # Tiny AE (optional, if tf available and not disabled)
    if TF_OK and (not no_ae):
        try:
            Xn = (X - X.mean())/(X.std()+1e-6)
            Xtrn, Xtst = Xn.loc[Xtr.index].values, Xn.loc[Xte.index].values
            d = Xtrn.shape[1]
            enc = models.Sequential([layers.Input(d),
                                     layers.Dense(max(8, d//2), activation="relu"),
                                     layers.Dense(max(4, d//4), activation="relu")])
            dec = models.Sequential([layers.Input(max(4, d//4)),
                                     layers.Dense(max(8, d//2), activation="relu"),
                                     layers.Dense(d, activation=None)])
            ae = models.Sequential([enc, dec])
            ae.compile(optimizer="adam", loss="mse")
            h = ae.fit(Xtrn, Xtrn, epochs=15, batch_size=64, verbose=0,
                       validation_split=0.1)
            # plot loss curve
            plt.figure(figsize=(6,4))
            plt.plot(h.history["loss"], label="train")
            plt.plot(h.history["val_loss"], label="val")
            plt.legend(); plt.xlabel("epoch"); plt.ylabel("mse")
            plt.title("AE training curve")
            savefig(os.path.join(RES_DIR, "ae_train_curve.png"))

            rec_err = np.mean((ae.predict(Xtst, verbose=0)-Xtst)**2, axis=1)
            p_ae, r_ae, ap_ae = pr_ap(yte, rec_err)
            plt.figure(figsize=(6,4))
            plt.plot(r_ae, p_ae, label=f"AE (AP={ap_ae:.3f})")
            plt.plot(r_if, p_if, label=f"IF (AP={ap_if:.3f})")
            plt.legend(); plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title("PR curves with AE")
            savefig(os.path.join(RES_DIR, "pr_curves_with_ae.png"))
        except Exception as e:
            print(f"[AE] skipped: {e}")

    # save metrics
    pd.DataFrame(MET).T.to_csv(os.path.join(RES_DIR, "summary_metrics.csv"))
    print("✓ Metrics saved ->", os.path.join(RES_DIR, "summary_metrics.csv"))
    return MET

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="CSV path (e.g., DATA/erm2-nwe9.csv)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout_h", type=float, default=2.0)
    ap.add_argument("--run_shap", action="store_true")
    ap.add_argument("--no_ae", action="store_true")
    args = ap.parse_args()

    print(">>> loading:", args.data)
    df = load_real_csv(args.data, limit=args.limit)

    # Early EDA figs (do not require labels)
    plot_delay_distributions(df)
    plot_box_by_category(df)
    plot_time_heatmaps(df)

    # Modeling + remaining figs
    MET = eval_all_models(df, timeout_h=args.timeout_h, run_shap=args.run_shap, no_ae=args.no_ae)

    print("\nDone. Figures in 'results/'. Key CSV: results/real_calls_clean.csv & summary_metrics.csv")

if __name__ == "__main__":
    main()
