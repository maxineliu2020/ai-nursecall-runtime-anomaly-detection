# experiment.py
# Runtime Anomaly Detection & Assurance for Nurse Call Systems
# Models: IF, Rule, OCSVM, RF, (optional) AE
# Outputs: results/summary_metrics.csv, pr_curves_multi.png, metrics_bar_ci_ar010.png, shap_rf_bar.png

import os, time, random, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support, precision_recall_curve, auc
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM

# ----------------------------
# Utils: seeds & dirs
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_out():
    os.makedirs("results", exist_ok=True)

# ----------------------------
# Data generation & anomalies
# ----------------------------
def generate_synthetic_logs(n_events=1000, start=None, rooms=50, emergency_ratio=0.20):
    """Generate base nurse-call like logs (normal)."""
    if start is None:
        start = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows, t0 = [], start
    for i in range(n_events):
        room = 100 + np.random.randint(rooms)
        is_emg = np.random.rand() < emergency_ratio
        delay = np.random.normal(loc=25 if is_emg else 45, scale=10)
        delay = max(1, delay)
        rows.append({
            "timestamp": (t0 + timedelta(seconds=i*5)).isoformat(),
            "room_id": room,
            "call_type": "emergency" if is_emg else "normal",
            "response_delay_s": float(delay),
            "label": 0
        })
    return pd.DataFrame(rows)

def inject_anomalies(df: pd.DataFrame, anomaly_rate=0.10):
    """Inject 5 anomaly types: delay, repeat, missing, timestamp, volume."""
    df = df.copy()
    n = len(df); k = max(1, int(n * anomaly_rate))
    idx = np.random.choice(df.index, size=k, replace=False)

    # some to remove (missing), some to mess timestamp
    remove_idx = set(np.random.choice(idx, size=max(1, k//10), replace=False))
    ts_idx = set(np.random.choice(list(set(idx)-remove_idx), size=max(1, k//10), replace=False))

    # the rest distributed among delay/repeat/volume
    other_idx = list(set(idx) - remove_idx - ts_idx)

    for i in other_idx:
        kind = random.choice(["delay","repeat","volume"])
        if kind == "delay":
            df.at[i, "response_delay_s"] = float(np.random.uniform(100, 240))
            df.at[i, "label"] = 1
        elif kind == "repeat":
            row = df.loc[i].copy()
            base_ts = datetime.fromisoformat(row["timestamp"])
            for j in range(np.random.randint(2,4)):
                new = row.copy()
                new["timestamp"] = (base_ts + timedelta(seconds=10*j)).isoformat()
                new["label"] = 1
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        elif kind == "volume":
            row = df.loc[i].copy()
            base_ts = datetime.fromisoformat(row["timestamp"])
            for j in range(10):
                new = row.copy()
                new["timestamp"] = (base_ts + timedelta(seconds=np.random.randint(0,60))).isoformat()
                new["label"] = 1
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

    if remove_idx:
        df = df.drop(index=list(remove_idx), errors="ignore")

    if ts_idx:
        for i in ts_idx:
            ts = datetime.fromisoformat(df.at[i, "timestamp"])
            df.at[i, "timestamp"] = (ts - timedelta(minutes=np.random.randint(5,30))).isoformat()
            df.at[i, "label"] = 1

    return df.sort_values("timestamp").reset_index(drop=True)

# ----------------------------
# Features
# ----------------------------
def featurize(df: pd.DataFrame):
    """Tabular features suitable for classic ML."""
    X = pd.DataFrame()
    X["delay"] = df["response_delay_s"].astype(float)
    X["is_emg"] = (df["call_type"]=="emergency").astype(int)
    ts = pd.to_datetime(df["timestamp"])
    X["minute"] = ts.dt.minute
    X["second"] = ts.dt.second
    X["room_mod"] = (df["room_id"].astype(int) % 10)
    return X

# ----------------------------
# Metrics, latency, alerts
# ----------------------------
def evaluate(y_true, y_pred, y_score=None):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"precision": p, "recall": r, "f1": f1}
    if y_score is not None and len(set(y_true))>1:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        out["pr_auc"] = auc(rec, prec)
        out["pr_curve"] = (prec, rec)
    return out

def rule_baseline(df):
    return (df["response_delay_s"] > 60).astype(int)

def measure_latency(model, X):
    t0 = time.time()
    _ = model.fit_predict(X)
    t1 = time.time()
    return (t1 - t0) / len(X)

def alerts_per_hour(pred, ts):
    s = pd.Series(pred, index=pd.to_datetime(ts))
    hourly = s.resample("1H").sum()
    return hourly.mean()

def mean_ci(arr):
    arr = np.array(arr, dtype=float)
    m = np.mean(arr)
    se = stats.sem(arr) if len(arr)>1 else 0.0
    ci95 = 1.96*se if len(arr)>1 else 0.0
    return m, (m-ci95, m+ci95)

# ----------------------------
# Extra models (OCSVM, RF, AE)
# ----------------------------
def run_ocsvm(X):
    mdl = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    mdl.fit(X)
    score = -mdl.decision_function(X)      # larger = more anomalous
    thr = np.median(score)
    y_pred = (score >= thr).astype(int)
    return y_pred, score

def run_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=0
    )
    rf.fit(X, y)
    prob = rf.predict_proba(X)[:,1]
    y_pred = (prob >= 0.5).astype(int)
    return y_pred, prob, rf

def run_autoencoder(X, y=None, epochs=8, batch_size=128, seed=0):
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        from tensorflow.keras import layers, models, optimizers

        dim = X.shape[1]
        inp = layers.Input(shape=(dim,))
        h = layers.Dense(16, activation='relu')(inp)
        h = layers.Dense(8, activation='relu')(h)
        z = layers.Dense(4, activation='relu')(h)
        h2 = layers.Dense(8, activation='relu')(z)
        h2 = layers.Dense(16, activation='relu')(h2)
        out = layers.Dense(dim, activation='linear')(h2)
        ae = models.Model(inp, out)
        ae.compile(optimizer=optimizers.Adam(1e-3), loss='mse')

        normal = X[(y==0)] if y is not None else X
        ae.fit(normal, normal, epochs=epochs, batch_size=batch_size, verbose=0)

        recon = ae.predict(X, verbose=0)
        mse = np.mean((X - recon)**2, axis=1)
        thr = np.median(mse)
        y_pred = (mse >= thr).astype(int)
        return y_pred, mse
    except Exception as e:
        print("[AE] skipped:", e)
        return None, None

# ----------------------------
# One run (all models)
# ----------------------------
def run_once(anomaly_rate=0.10, seed=0, use_ae=True):
    set_seed(seed)
    df = generate_synthetic_logs(n_events=1000)
    df = inject_anomalies(df, anomaly_rate)
    Xdf = featurize(df)
    X = Xdf.values
    y = df["label"].astype(int).values

    # IF
    if_mdl = IsolationForest(n_estimators=200, contamination='auto', random_state=seed)
    if_scores = -if_mdl.fit(X).decision_function(X)
    if_thr = np.median(if_scores)
    if_pred = (if_scores >= if_thr).astype(int)
    m_if = evaluate(y, if_pred, y_score=if_scores)

    # Rule
    y_rule = rule_baseline(df)
    m_rule = evaluate(y, y_rule)

    # OCSVM
    oc_pred, oc_score = run_ocsvm(X)
    m_oc = evaluate(y, oc_pred, y_score=oc_score)

    # RF (supervised)
    rf_pred, rf_prob, rf_model = run_rf(X, y)
    m_rf = evaluate(y, rf_pred, y_score=rf_prob)

    # AE (optional)
    if use_ae:
        ae_pred, ae_score = run_autoencoder(X, y, epochs=8, seed=seed)
        m_ae = evaluate(y, ae_pred, y_score=ae_score) if ae_pred is not None else None
    else:
        m_ae = None

    # Latency & alerts (using IF as reference)
    lat_if = measure_latency(IsolationForest(n_estimators=200, random_state=seed), X)
    aph_if = alerts_per_hour(if_pred, df["timestamp"])

    return {
        "df": df, "X": X, "feature_names": list(Xdf.columns),
        "rf_model": rf_model,
        "metrics": {"IF": m_if, "Rule": m_rule, "OCSVM": m_oc, "RF": m_rf, "AE": m_ae},
        "latency_s_per_record": lat_if, "alerts_per_hour": aph_if
    }

# ----------------------------
# Plots
# ----------------------------
def save_multi_pr_curves(curves_by_model, fname="results/pr_curves_multi.png"):
    plt.figure(figsize=(6,4))
    for model_name, (prec, rec) in curves_by_model.items():
        plt.plot(rec, prec, label=model_name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (last run)")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(fname, dpi=180); plt.close()

def save_bar_ci_ar(summary_df, ar=0.10, fname="results/metrics_bar_ci_ar010.png"):
    if ar not in summary_df["anomaly_rate"].values:
        return
    row = summary_df.loc[summary_df["anomaly_rate"]==ar].iloc[0]

    models = ["IF","Rule","OCSVM","RF","AE"]
    labels = ["Precision","Recall","F1"]
    x = np.arange(len(labels))
    width = 0.14

    fig, ax = plt.subplots(figsize=(7.5,4.5))
    offs = np.linspace(-width*2, width*2, num=len(models))

    for idx, m in enumerate(models):
        if f"{m}_prec_mean" not in row: 
            continue
        means = [row[f"{m}_prec_mean"], row[f"{m}_rec_mean"], row[f"{m}_f1_mean"]]
        lo = [row[f"{m}_prec_ci95_low"], row[f"{m}_rec_ci95_low"], row[f"{m}_f1_ci95_low"]]
        hi = [row[f"{m}_prec_ci95_high"], row[f"{m}_rec_ci95_high"], row[f"{m}_f1_ci95_high"]]
        yerr = [[means[i]-lo[i] for i in range(3)], [hi[i]-means[i] for i in range(3)]]
        ax.bar(x + offs[idx], means, width, yerr=yerr, capsize=3, label=m)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics at AR={ar} (mean ± 95% CI)")
    ax.legend(ncol=3, fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(fname, dpi=180); plt.close()

def save_shap_bar(rf_model, X, feature_names, fname="results/shap_rf_bar.png"):
    try:
        import shap
        expl = shap.TreeExplainer(rf_model)
        vals = expl.shap_values(X)
        shap.summary_plot(vals[1], pd.DataFrame(X, columns=feature_names),
                          plot_type="bar", show=False, max_display=10)
        plt.tight_layout(); plt.savefig(fname, dpi=180); plt.close()
    except Exception as e:
        print("[SHAP] skipped:", e)

# ----------------------------
# Main loop
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly_rates", type=str, default="0.05,0.10,0.20",
                        help="comma-separated anomaly rates")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--no-ae", action="store_true", help="skip autoencoder")
    args = parser.parse_args()

    ensure_out()
    ar_list = [float(x) for x in args.anomaly_rates.split(",")]
    repeats = args.repeats
    base_seed = args.seed
    use_ae = not args.no_ae

    model_list = ["IF","Rule","OCSVM","RF","AE" if use_ae else ""]

    summary_rows = []
    last_curves = {}  # for multi-model PR curves (from last run)

    for ar in ar_list:
        stats = {m: {"prec":[], "rec":[], "f1":[], "prauc":[]} for m in ["IF","Rule","OCSVM","RF","AE"]}

        for r in tqdm(range(repeats), desc=f"AR={ar}"):
            res = run_once(anomaly_rate=ar, seed=base_seed+r, use_ae=use_ae)
            mets = res["metrics"]
            for mname, m in mets.items():
                if m is None: 
                    continue
                stats[mname]["prec"].append(m["precision"])
                stats[mname]["rec"].append(m["recall"])
                stats[mname]["f1"].append(m["f1"])
                if "pr_curve" in m:
                    last_curves[mname] = m["pr_curve"]
                if "pr_auc" in m:
                    stats[mname]["prauc"].append(m["pr_auc"])

            last_result = res  # keep last for SHAP/latency
        # pack results
        row = {"anomaly_rate": ar}
        for m in ["IF","Rule","OCSVM","RF","AE"]:
            for key, alias in [("prec","prec"), ("rec","rec"), ("f1","f1"), ("prauc","pr_auc")]:
                arr = stats[m][key]
                if len(arr)==0: 
                    continue
                mean, (lo, hi) = mean_ci(arr)
                row[f"{m}_{alias}_mean"] = mean
                row[f"{m}_{alias}_ci95_low"] = lo
                row[f"{m}_{alias}_ci95_high"] = hi

        # latency & alerts (use IF reference)
        lat = last_result["latency_s_per_record"]
        aph = last_result["alerts_per_hour"]
        mean_lat, (lo_lat, hi_lat) = mean_ci([lat])
        mean_aph, (lo_aph, hi_aph) = mean_ci([aph])
        row["latency_s_per_record_mean"] = mean_lat
        row["latency_s_per_record_ci95_low"] = lo_lat
        row["latency_s_per_record_ci95_high"] = hi_lat
        row["alerts_per_hour_mean"] = mean_aph
        row["alerts_per_hour_ci95_low"] = lo_aph
        row["alerts_per_hour_ci95_high"] = hi_aph

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/summary_metrics.csv", index=False)
    print("\nSaved: results/summary_metrics.csv")

    # Multi-model PR curves (last run)
    if last_curves:
        save_multi_pr_curves(last_curves, "results/pr_curves_multi.png")
        print("Saved: results/pr_curves_multi.png")

    # Bar chart with CI at AR=0.10 (or first available)
    target_ar = 0.10 if 0.10 in summary["anomaly_rate"].values else summary["anomaly_rate"].iloc[0]
    save_bar_ci_ar(summary, ar=target_ar, fname="results/metrics_bar_ci_ar010.png")
    print("Saved: results/metrics_bar_ci_ar010.png")

    # SHAP for RF (last run)
    save_shap_bar(last_result["rf_model"], last_result["X"], last_result["feature_names"],
                  "results/shap_rf_bar.png")
    print("Saved: results/shap_rf_bar.png")

if __name__ == "__main__":
    main()
