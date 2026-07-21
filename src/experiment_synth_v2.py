# -*- coding: utf-8 -*-
# =============================================================================
# experiment_synth_v2.py — Pipeline A (supplementary): synthetic anomaly injection
#
# Part of the reproducibility package for:
#   "Simulation-Based Runtime Anomaly Detection for Nurse Call System Assurance:
#    A Reproducible Proof of Concept Using Public Service-Request Logs"
#
# Authors:
#   Yuanyuan (Maxine) Liu (1,2)  — corresponding author
#       yliu536@jh.edu | yl366@mynsu.nova.edu | ORCID 0000-0003-3410-6893
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
experiment_synth_v2.py — Corrected evaluation protocol for the synthetic-injection
pipeline (pipeline A). Injection design is kept faithful to the archived v1.0.1
experiment.py (same distributions and parameters) so that the five-anomaly-type
narrative is supported by a methodologically sound supplementary experiment.

Fixes vs v1 experiment.py:
  1. 60/20/20 train/val/test split (stratified); v1 fit and evaluated on the same data.
  2. Thresholds selected on VALIDATION only; v1 used the score median (forced ~50% alert rate).
  3. Fresh independent dataset per seed (true independent runs).
  4. Full metrics incl. AP, AUC-ROC, FPR, FNR, alert rate + per-anomaly-type recall.
  5. Inference-only latency (v1 timed fit_predict, i.e. training included).
  6. Honest handling of 'missing' anomalies: removed rows are unobservable to a
     row-wise detector and are reported as injected-but-not-row-evaluable.
"""
import os, sys, time, json, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix)

OUT = "/home/claude/synth_v2_results"
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(201, 211))
RATES = [0.05, 0.10, 0.20]
N_EVENTS = 1000  # same as v1

# ---------- generation & injection (faithful to v1 parameters) ----------
def generate(seed, n=N_EVENTS, rooms=50, emg_ratio=0.20):
    rng = np.random.default_rng(seed)
    t0 = datetime(2025, 1, 6, 8, 0, 0)
    rows = []
    for i in range(n):
        is_emg = rng.random() < emg_ratio
        delay = max(1.0, rng.normal(25 if is_emg else 45, 10))
        rows.append(dict(timestamp=t0 + timedelta(seconds=i * 5),
                         room_id=100 + int(rng.integers(rooms)),
                         call_type="emergency" if is_emg else "normal",
                         response_delay_s=float(delay), label=0, atype="normal"))
    return pd.DataFrame(rows), rng

def inject(df, rate, rng):
    df = df.copy()
    n = len(df); k = max(1, int(n * rate))
    idx = rng.choice(df.index.to_numpy(), size=k, replace=False)
    n_side = max(1, k // 10)
    remove_idx = set(rng.choice(idx, size=n_side, replace=False))
    rest = np.array(sorted(set(idx) - remove_idx))
    ts_idx = set(rng.choice(rest, size=min(n_side, len(rest)), replace=False))
    other = sorted(set(idx) - remove_idx - ts_idx)
    extra = []
    for i in other:
        kind = ["delay", "repeat", "volume"][int(rng.integers(3))]
        if kind == "delay":
            df.at[i, "response_delay_s"] = float(rng.uniform(100, 240))
            df.at[i, "label"] = 1; df.at[i, "atype"] = "delay"
        elif kind == "repeat":
            base = df.loc[i].copy()
            for j in range(int(rng.integers(2, 4))):
                new = base.copy(); new["timestamp"] = base["timestamp"] + timedelta(seconds=10 * j)
                new["label"] = 1; new["atype"] = "repeat"; extra.append(new)
        else:  # volume burst: 10 calls within one minute
            base = df.loc[i].copy()
            for j in range(10):
                new = base.copy()
                new["timestamp"] = base["timestamp"] + timedelta(seconds=int(rng.integers(0, 60)))
                new["label"] = 1; new["atype"] = "volume"; extra.append(new)
    n_missing = len(remove_idx)
    df = df.drop(index=list(remove_idx))
    for i in ts_idx:
        df.at[i, "timestamp"] = df.at[i, "timestamp"] - timedelta(minutes=int(rng.integers(5, 30)))
        df.at[i, "label"] = 1; df.at[i, "atype"] = "timestamp"
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, n_missing

def featurize(df):
    X = pd.DataFrame(index=df.index)
    X["delay"] = df["response_delay_s"].astype(float)
    X["is_emg"] = (df["call_type"] == "emergency").astype(int)
    ts = pd.to_datetime(df["timestamp"])
    X["minute"] = ts.dt.minute
    X["second"] = ts.dt.second
    X["room_mod"] = df["room_id"].astype(int) % 10
    return X

# ---------- eval helpers ----------
def pick_threshold_val(score_val, y_val):
    qs = np.unique(np.percentile(score_val, np.linspace(1, 99.5, 200)))
    best_t, best_f1 = qs[0], -1
    for t in qs:
        f = f1_score(y_val, (score_val >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return best_t

def metrics(y, yhat, score, atypes):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = dict(precision=precision_score(y, yhat, zero_division=0),
               recall=recall_score(y, yhat, zero_division=0),
               f1=f1_score(y, yhat, zero_division=0),
               fpr=fp / (fp + tn) if fp + tn else 0.0,
               fnr=fn / (fn + tp) if fn + tp else 0.0,
               alert_rate=float(np.mean(yhat)))
    if score is not None and len(np.unique(y)) > 1:
        out["ap"] = average_precision_score(y, score)
        out["auc_roc"] = roc_auc_score(y, score)
    for t in ["delay", "repeat", "volume", "timestamp"]:
        m = (atypes == t)
        if m.sum() > 0:
            out[f"recall_{t}"] = float(np.mean(yhat[m]))
    return out

def mean_ci(a):
    a = np.asarray(a, float); m = a.mean()
    h = stats.t.ppf(0.975, len(a) - 1) * stats.sem(a) if len(a) > 1 else np.nan
    return m, m - h, m + h, (a.std(ddof=1) if len(a) > 1 else 0.0)

def aggregate(rows):
    models = sorted({m for r in rows for m in r})
    recs = []
    for m in models:
        rec = {"model": m}
        keys = sorted({k for r in rows if m in r for k in r[m]})
        for k in keys:
            vals = [r[m][k] for r in rows if m in r and k in r[m]]
            mu, lo, hi, sd = mean_ci(vals)
            rec[f"{k}_mean"] = mu; rec[f"{k}_sd"] = sd
            rec[f"{k}_ci_lo"] = lo; rec[f"{k}_ci_hi"] = hi
        recs.append(rec)
    return pd.DataFrame(recs)

# ---------- one run ----------
def one_run(rate, seed):
    df0, rng = generate(seed)
    df, n_missing = inject(df0, rate, rng)
    X = featurize(df); y = df["label"].to_numpy(); at = df["atype"].to_numpy()
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])
    res = {}
    ifo = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X.iloc[tr])
    sv, st_ = -ifo.score_samples(X.iloc[va]), -ifo.score_samples(X.iloc[te])
    t = pick_threshold_val(sv, y[va])
    res["IF"] = metrics(y[te], (st_ >= t).astype(int), st_, at[te])
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X.iloc[tr])
    sv, st_ = -oc.score_samples(X.iloc[va]), -oc.score_samples(X.iloc[te])
    t = pick_threshold_val(sv, y[va])
    res["OCSVM"] = metrics(y[te], (st_ >= t).astype(int), st_, at[te])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(X.iloc[tr], y[tr])
    pv, pt = rf.predict_proba(X.iloc[va])[:, 1], rf.predict_proba(X.iloc[te])[:, 1]
    t = pick_threshold_val(pv, y[va])
    res["RF"] = metrics(y[te], (pt >= t).astype(int), pt, at[te])
    # Rule baseline (>60s), independent of the injection mechanism
    yr = (df["response_delay_s"].to_numpy()[te] > 60).astype(int)
    res["Rule60s"] = metrics(y[te], yr, None, at[te])
    return res, dict(n=len(df), n_te=len(te), prevalence=float(y.mean()),
                     n_missing_injected_unobservable=n_missing)

def latency_bench(seed=201, reps=200):
    df0, rng = generate(seed); df, _ = inject(df0, 0.10, rng)
    X = featurize(df)
    ifo = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
    one = X.iloc[[0]]
    lat = []
    for _ in range(reps):
        t0 = time.perf_counter(); ifo.score_samples(one); lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    t0 = time.perf_counter(); ifo.score_samples(X); bt = time.perf_counter() - t0
    return dict(single_ms_mean=float(lat.mean()), single_ms_p95=float(np.percentile(lat, 95)),
                batch_events_per_s=len(X) / bt,
                model_size_mb=len(pickle.dumps(ifo)) / 1e6)

def main():
    all_meta = []
    for rate in RATES:
        rows = []
        for sd in SEEDS:
            r, meta = one_run(rate, sd); rows.append(r); meta["rate"] = rate
            all_meta.append(meta)
        agg = aggregate(rows); agg.insert(0, "anomaly_rate", rate)
        f = f"{OUT}/synth_metrics.csv"
        agg.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        print(f"rate={rate} done; mean prevalence="
              f"{np.mean([m['prevalence'] for m in all_meta if m['rate']==rate]):.3f}")
    json.dump(all_meta, open(f"{OUT}/run_meta.json", "w"), indent=2)
    json.dump(latency_bench(), open(f"{OUT}/latency.json", "w"), indent=2)
    print("done ->", OUT)

if __name__ == "__main__":
    main()
