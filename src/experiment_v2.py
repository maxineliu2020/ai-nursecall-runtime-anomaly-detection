# -*- coding: utf-8 -*-
# =============================================================================
# experiment_v2.py — Pipeline B (primary): NYC 311 transformation experiments
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
experiment_v2.py — Revised protocol for Scientific Reports resubmission.
Runtime Anomaly Detection for Nurse-Call-like Service Logs (NYC 311 proxy).

Fixes relative to experiment_real_plus.py (v1.0.1):
  1. Explicit, reported data cleaning (negative / missing / zero response intervals).
  2. Records with missing closure are EXCLUDED from delay-label evaluation
     (right-censored; previously silently labeled normal).
  3. 60/20/20 train/validation/test split; thresholds selected on VALIDATION only.
  4. >=10 independent seeded runs; t-interval 95% CIs over runs.
  5. Leakage ablation: feature set A (with resp_h) vs B (without resp_h).
  6. Independent rule baseline (per-category train quantile), distinct in form
     from the global-timeout label rule; plus label-free alert-budget thresholds.
  7. Full metrics: P, R, F1, AP, AUC-ROC, FPR, FNR, specificity, alert rate,
     alerts/hour over the test time window.
  8. Timeout (label-definition) sweep and fixed-prevalence subsampling.
  9. Runtime benchmarks: single-event latency, batch throughput, model size.
 10. Environment versions recorded.
"""
import os, sys, time, json, pickle, platform
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix)

DATA = "/mnt/user-data/uploads/erm2-nwe9.csv"
OUT = "/home/claude/v2_results"
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(101, 111))          # 10 independent seeds
TIMEOUT_PRIMARY = 2.0                   # retained for comparability with v1
TIMEOUT_SWEEP = [1.0, 2.0, 6.0, 12.0, 24.0, 48.0]
PREV_TARGETS = [0.01, 0.02, 0.05, 0.10]

# ---------------- data ----------------
def load_clean():
    df = pd.read_csv(DATA, low_memory=False)
    n_raw = len(df)
    df["_created"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["_closed"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df["_category"] = df["agency"].astype(str)
    df["resp_h"] = (df["_closed"] - df["_created"]).dt.total_seconds() / 3600.0
    n_missing = int(df["resp_h"].isna().sum())
    n_neg = int((df["resp_h"] < 0).sum())
    n_zero = int((df["resp_h"] == 0).sum())
    n_gt7d = int((df["resp_h"] > 168).sum())
    # cleaning: drop missing closure (right-censored) and negative (invalid timestamps)
    keep = df["resp_h"].notna() & (df["resp_h"] >= 0)
    dfc = df.loc[keep].sort_values("_created").reset_index(drop=True)
    audit = dict(n_raw=n_raw, n_missing_closure=n_missing, n_negative=n_neg,
                 n_zero=n_zero, n_gt_7d=n_gt7d, n_retained=len(dfc),
                 created_min=str(df["_created"].min()), created_max=str(df["_created"].max()))
    return dfc, audit

def featurize(df, include_resp=True, top_cats=None):
    X = pd.DataFrame(index=df.index)
    X["hour"] = df["_created"].dt.hour
    X["weekday"] = df["_created"].dt.weekday
    if include_resp:
        X["resp_h"] = df["resp_h"]
    if top_cats is None:
        top_cats = df["_category"].value_counts().head(8).index.tolist()
    for c in top_cats:
        X[f"cat_{c}"] = (df["_category"] == c).astype(int)
    return X.fillna(0), top_cats

# ---------------- metrics ----------------
def full_metrics(y, yhat, score, hours_span):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = dict(
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        fpr=fp / (fp + tn) if (fp + tn) else 0.0,
        fnr=fn / (fn + tp) if (fn + tp) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        alert_rate=float(np.mean(yhat)),
        alerts_per_hour=float(np.sum(yhat)) / hours_span if hours_span > 0 else np.nan,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )
    if score is not None and len(np.unique(y)) > 1:
        out["ap"] = average_precision_score(y, score)
        out["auc_roc"] = roc_auc_score(y, score)
    return out

def pick_threshold_val(score_val, y_val):
    """Label-informed retrospective threshold: maximize F1 on VALIDATION."""
    qs = np.percentile(score_val, np.linspace(1, 99.5, 300))
    best_t, best_f1 = qs[0], -1
    for t in np.unique(qs):
        f1v = f1_score(y_val, (score_val >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_t = f1v, t
    return best_t

def mean_ci(a):
    a = np.asarray(a, float)
    m = a.mean()
    if len(a) > 1:
        h = stats.t.ppf(0.975, len(a) - 1) * stats.sem(a)
    else:
        h = np.nan
    return m, m - h, m + h, a.std(ddof=1) if len(a) > 1 else 0.0

# ---------------- single run ----------------
def one_run(dfc, timeout_h, seed, include_resp=True, prevalence_target=None, run_ocsvm=True, ocsvm_cap=15000):
    df = dfc.copy()
    df["is_anomaly"] = (df["resp_h"] > timeout_h).astype(int)
    if prevalence_target is not None:
        rng = np.random.default_rng(seed)
        pos = df.index[df["is_anomaly"] == 1].to_numpy()
        neg = df.index[df["is_anomaly"] == 0].to_numpy()
        n_pos_keep = int(round(prevalence_target * len(neg) / (1 - prevalence_target)))
        n_pos_keep = min(n_pos_keep, len(pos))
        keep = np.concatenate([rng.choice(pos, n_pos_keep, replace=False), neg])
        df = df.loc[np.sort(keep)].reset_index(drop=True)
    y = df["is_anomaly"].to_numpy()
    X, top_cats = featurize(df, include_resp=include_resp)

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=y[tmp])
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    span_h = (df["_created"].iloc[te].max() - df["_created"].iloc[te].min()).total_seconds() / 3600.0

    res = {}
    # Isolation Forest
    ifo = IsolationForest(n_estimators=400, random_state=seed, n_jobs=-1,
                          contamination="auto").fit(Xtr)
    s_va, s_te = -ifo.score_samples(Xva), -ifo.score_samples(Xte)
    t = pick_threshold_val(s_va, yva)
    res["IF"] = full_metrics(yte, (s_te >= t).astype(int), s_te, span_h)
    # label-free alert budgets (deployment view), IF only
    for q in (0.90, 0.80):
        tq = np.quantile(np.concatenate([-ifo.score_samples(Xtr)]), q)
        res[f"IF_budget_top{int((1-q)*100)}pct"] = full_metrics(
            yte, (s_te >= tq).astype(int), None, span_h)

    # One-Class SVM (trained on a seeded random subsample of <=ocsvm_cap
    # training records for computational tractability; documented protocol note)
    if run_ocsvm:
        rng_oc = np.random.default_rng(seed)
        sub = rng_oc.choice(len(Xtr), size=min(ocsvm_cap, len(Xtr)), replace=False)
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr.iloc[sub])
        s_va, s_te = -oc.score_samples(Xva), -oc.score_samples(Xte)
        t = pick_threshold_val(s_va, yva)
        res["OCSVM"] = full_metrics(yte, (s_te >= t).astype(int), s_te, span_h)

    # Random Forest (supervised reference / upper bound)
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    p_va, p_te = rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1]
    t = pick_threshold_val(p_va, yva)
    res["RF"] = full_metrics(yte, (p_te >= t).astype(int), p_te, span_h)

    # Independent rule baseline: per-category train p80 of resp_h
    # (different functional form from the global-timeout label rule)
    if include_resp:
        thr_map = df.iloc[tr].groupby("_category")["resp_h"].quantile(0.80)
        global_thr = df.iloc[tr]["resp_h"].quantile(0.80)
        thr_te = df.iloc[te]["_category"].map(thr_map).fillna(global_thr).to_numpy()
        yhat = (df.iloc[te]["resp_h"].to_numpy() > thr_te).astype(int)
        res["RuleCat80"] = full_metrics(yte, yhat, None, span_h)

    return res, dict(n=len(df), n_tr=len(tr), n_va=len(va), n_te=len(te),
                     prevalence=float(y.mean()), models=ifo if False else None)

# ---------------- aggregation ----------------
def aggregate(rows):
    """rows: list of dict[model] -> metrics dict"""
    models = sorted({m for r in rows for m in r})
    table = []
    for m in models:
        rec = {"model": m, "n_runs": sum(1 for r in rows if m in r)}
        keys = sorted({k for r in rows if m in r for k in r[m] if not isinstance(r[m][k], int)})
        for k in keys:
            vals = [r[m][k] for r in rows if m in r and k in r[m] and r[m][k] == r[m][k]]
            if not vals:
                continue
            mean, lo, hi, sd = mean_ci(vals)
            rec[f"{k}_mean"] = mean; rec[f"{k}_sd"] = sd
            rec[f"{k}_ci_lo"] = lo; rec[f"{k}_ci_hi"] = hi
        table.append(rec)
    return pd.DataFrame(table)

# ---------------- runtime benchmark ----------------
def runtime_benchmark(dfc, seed=101, reps=200):
    df = dfc.copy()
    df["is_anomaly"] = (df["resp_h"] > TIMEOUT_PRIMARY).astype(int)
    X, _ = featurize(df, include_resp=True)
    y = df["is_anomaly"].to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    out = {}
    for name, mdl in [("IF", IsolationForest(n_estimators=400, random_state=seed, n_jobs=-1)),
                      ("RF", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1))]:
        t0 = time.perf_counter()
        mdl.fit(Xtr, y[tr]) if name == "RF" else mdl.fit(Xtr)
        fit_s = time.perf_counter() - t0
        score = (lambda Z: mdl.predict_proba(Z)[:, 1]) if name == "RF" else (lambda Z: -mdl.score_samples(Z))
        # batch throughput
        t0 = time.perf_counter(); score(Xte); batch_s = time.perf_counter() - t0
        # single-event latency
        one = Xte.iloc[[0]]
        lat = []
        for _ in range(reps):
            t0 = time.perf_counter(); score(one); lat.append(time.perf_counter() - t0)
        lat = np.array(lat) * 1000.0
        size_mb = len(pickle.dumps(mdl)) / 1e6
        out[name] = dict(fit_s=fit_s, batch_events=len(Xte), batch_s=batch_s,
                         events_per_s=len(Xte) / batch_s,
                         single_latency_ms_mean=float(lat.mean()),
                         single_latency_ms_sd=float(lat.std(ddof=1)),
                         single_latency_ms_p95=float(np.percentile(lat, 95)),
                         model_size_mb=size_mb)
    out["hardware"] = dict(platform=platform.platform(), machine=platform.machine(),
                           cpu_count=os.cpu_count(), python=sys.version.split()[0])
    return out

# ---------------- main ----------------
def main():
    dfc, audit = load_clean()
    json.dump(audit, open(f"{OUT}/data_audit.json", "w"), indent=2)
    print("AUDIT:", audit)

    # env versions
    import sklearn
    env = dict(python=sys.version.split()[0], numpy=np.__version__,
               pandas=pd.__version__, sklearn=sklearn.__version__,
               scipy=__import__("scipy").__version__)
    json.dump(env, open(f"{OUT}/environment.json", "w"), indent=2)

    # ---- Main experiment: timeout=2.0, features A (with resp_h), 10 seeds
    rowsA, rowsB, meta_list = [], [], []
    for sd in SEEDS:
        rA, meta = one_run(dfc, TIMEOUT_PRIMARY, sd, include_resp=True)
        rB, _ = one_run(dfc, TIMEOUT_PRIMARY, sd, include_resp=False, run_ocsvm=False)
        rowsA.append(rA); rowsB.append(rB); meta_list.append(meta)
        print(f"seed {sd}: IF-A F1={rA['IF']['f1']:.3f} AP={rA['IF']['ap']:.3f} | "
              f"IF-B F1={rB['IF']['f1']:.3f} AP={rB['IF']['ap']:.3f}")
    aggregate(rowsA).to_csv(f"{OUT}/main_expA_with_resp.csv", index=False)
    aggregate(rowsB).to_csv(f"{OUT}/main_expB_no_resp.csv", index=False)
    json.dump(meta_list[0], open(f"{OUT}/split_meta.json", "w"), indent=2, default=str)

    # ---- Timeout sweep (label-definition sensitivity), IF + RF, 5 seeds
    sweep_rows = []
    for th in TIMEOUT_SWEEP:
        prev = float((dfc["resp_h"] > th).mean())
        runs = []
        for sd in SEEDS[:5]:
            r, _ = one_run(dfc, th, sd, include_resp=True, run_ocsvm=False)
            runs.append({k: r[k] for k in ("IF", "RF", "RuleCat80") if k in r})
        agg = aggregate(runs); agg.insert(0, "timeout_h", th); agg.insert(1, "prevalence", prev)
        sweep_rows.append(agg)
        print(f"timeout={th}h prevalence={prev:.3f} done")
    pd.concat(sweep_rows).to_csv(f"{OUT}/timeout_sweep.csv", index=False)

    # ---- Fixed-prevalence subsampling at timeout=2.0, IF, 5 seeds
    prev_rows = []
    for pt in PREV_TARGETS:
        runs = []
        for sd in SEEDS[:5]:
            r, meta = one_run(dfc, TIMEOUT_PRIMARY, sd, include_resp=True, prevalence_target=pt, run_ocsvm=False)
            runs.append({"IF": r["IF"], "RF": r["RF"]})
        agg = aggregate(runs); agg.insert(0, "prevalence_target", pt)
        prev_rows.append(agg)
        print(f"prevalence target={pt} done (n={meta['n']}, actual prev={meta['prevalence']:.4f})")
    pd.concat(prev_rows).to_csv(f"{OUT}/prevalence_sensitivity.csv", index=False)

    # ---- Runtime benchmark
    bench = runtime_benchmark(dfc)
    json.dump(bench, open(f"{OUT}/runtime_benchmark.json", "w"), indent=2)
    print("BENCH:", json.dumps(bench, indent=2)[:400])
    print("\nAll outputs in", OUT)

if __name__ == "__main__":
    main()
