# -*- coding: utf-8 -*-
# =============================================================================
# make_figures_v2.py — Regenerates all manuscript figures (300 dpi)
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
"""make_figures_v2.py — Regenerate all main + supplementary figures from the v2 protocol.
Fixes reviewer-flagged issues: Actual/Predicted axis labels, units, thresholds scale,
311 source-category labeling, error bars from real independent-run CIs.
Outputs 300-dpi PNGs to /home/claude/figures_v2/.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, confusion_matrix

import sys
sys.path.insert(0, "/home/claude")
import experiment_v2 as e2

FIG = "/home/claude/figures_v2"
os.makedirs(FIG, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})
SEED = 101

def save(name):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close()

# ---------------- data & seed-101 artifacts ----------------
dfc, audit = e2.load_clean()
df = dfc.copy()
df["is_anomaly"] = (df["resp_h"] > 2.0).astype(int)
y = df["is_anomaly"].to_numpy()
XA, cats = e2.featurize(df, include_resp=True)
XB, _ = e2.featurize(df, include_resp=False)
idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=y[tmp])

def fit_scores(X):
    ifo = IsolationForest(n_estimators=400, random_state=SEED, n_jobs=-1,
                          contamination="auto").fit(X.iloc[tr])
    return ifo, -ifo.score_samples(X.iloc[va]), -ifo.score_samples(X.iloc[te])

ifoA, svA, stA = fit_scores(XA)
ifoB, svB, stB = fit_scores(XB)
rng = np.random.default_rng(SEED)
sub = rng.choice(len(tr), 15000, replace=False)
oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(XA.iloc[tr].iloc[sub])
sv_oc, st_oc = -oc.score_samples(XA.iloc[va]), -oc.score_samples(XA.iloc[te])
rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(XA.iloc[tr], y[tr])
pv_rf, pt_rf = rf.predict_proba(XA.iloc[va])[:, 1], rf.predict_proba(XA.iloc[te])[:, 1]
rfB = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(XB.iloc[tr], y[tr])
ptB_rf = rfB.predict_proba(XB.iloc[te])[:, 1]
yte, yva = y[te], y[va]

# ================ Fig 1: architecture ================
fig, ax = plt.subplots(figsize=(7.0, 1.6))
boxes = ["311 log ingestion\n& transformation", "Delay-label\nconstruction",
         "Anomaly detection\n(IF / OCSVM / RF / rule)", "Evaluation &\nvisualization"]
for i, b in enumerate(boxes):
    ax.add_patch(mpl.patches.FancyBboxPatch((i * 1.9, 0), 1.55, 0.9,
                 boxstyle="round,pad=0.08", fc="#dbe9f6", ec="#3b6ea5", lw=1.2))
    ax.text(i * 1.9 + 0.78, 0.45, b, ha="center", va="center", fontsize=8.5)
    if i < 3:
        ax.annotate("", xy=(i * 1.9 + 1.83, 0.45), xytext=(i * 1.9 + 1.62, 0.45),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#3b6ea5"))
ax.set_xlim(-0.15, 7.5); ax.set_ylim(-0.15, 1.05); ax.axis("off")
save("Fig1_architecture")

# ================ Fig 2: weekday-hour heatmap ================
d = dfc.copy(); d["weekday"] = d["_created"].dt.weekday; d["hour"] = d["_created"].dt.hour
mat = d.pivot_table(index="weekday", columns="hour", values="resp_h", aggfunc="count").fillna(0)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
plt.colorbar(im, label="Record count")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Weekday (0 = Monday)")
ax.set_xticks(range(0, 24, 2)); ax.set_yticks(range(7))
ax.set_title("Transformed 311 service-log workload (proxy data)")
save("Fig2_heatmap_weekday_hour")

# ================ Fig 3: main metrics bar + 95% CI ================
A = pd.read_csv("/home/claude/v2_results/main_expA_with_resp.csv").set_index("model")
models = ["IF", "OCSVM", "RF", "RuleCat80"]
labels = ["Isolation\nForest", "One-Class\nSVM", "Random Forest\n(supervised ref.)", "Rule baseline\n(cat. p80)"]
mets = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
x = np.arange(len(models)); w = 0.25
fig, ax = plt.subplots(figsize=(6.6, 3.2))
for k, (m, lab) in enumerate(mets):
    mu = [A.loc[mm, f"{m}_mean"] for mm in models]
    lo = [mu[i] - A.loc[mm, f"{m}_ci_lo"] for i, mm in enumerate(models)]
    hi = [A.loc[mm, f"{m}_ci_hi"] - mu[i] for i, mm in enumerate(models)]
    ax.bar(x + (k - 1) * w, mu, w, yerr=[lo, hi], capsize=3, label=lab)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Detection performance, timeout = 2.0 h (mean ± 95% CI, 10 seeds)")
ax.legend(ncol=3); ax.grid(axis="y", alpha=0.3)
save("Fig3_metrics_bar_ci")

# ================ Fig 4: PR curves (seed 101) ================
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, sc in [("Isolation Forest", stA), ("One-Class SVM", st_oc), ("Random Forest (sup.)", pt_rf)]:
    p, r, _ = precision_recall_curve(yte, sc)
    ax.plot(r, p, label=f"{name} (AP={average_precision_score(yte, sc):.3f})")
ax.axhline(yte.mean(), ls="--", c="grey", lw=1, label=f"Chance (prevalence={yte.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision–recall curves, test set (seed 101)")
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
save("Fig4_pr_curves")

# ================ Fig 5: SHAP for RF (v2) ================
import shap
bg = XA.iloc[tr].sample(1000, random_state=SEED)
expl = shap.TreeExplainer(rf, data=bg, model_output="probability")
Xs = XA.iloc[te].sample(2000, random_state=SEED)
sv = expl.shap_values(Xs, check_additivity=False)
sv1 = sv[:, :, 1] if sv.ndim == 3 else sv
plt.figure(figsize=(5.6, 3.6))
shap.summary_plot(sv1, Xs, plot_type="bar", show=False, max_display=11)
plt.xlabel("Mean |SHAP value| (contribution to anomaly probability)")
plt.title("RF feature attribution (supervised reference, v2 protocol)", fontsize=10)
save("Fig5_shap_rf_bar")
json.dump({"shap_version": shap.__version__, "explainer": "TreeExplainer",
           "model": "RandomForestClassifier (supervised reference)",
           "background": "1000 training records (seeded sample)",
           "explained": "2000 test records (seeded sample)",
           "output": "probability of anomaly class"},
          open(f"{FIG}/shap_metadata.json", "w"), indent=2)

# ================ Fig 6: threshold sensitivity (IF, test) ================
qs = np.linspace(1, 99.5, 120)
prec, rec, f1s, alert = [], [], [], []
for q in qs:
    t = np.percentile(stA, q); yh = (stA >= t).astype(int)
    tp = ((yh == 1) & (yte == 1)).sum(); fp = ((yh == 1) & (yte == 0)).sum()
    fn = ((yh == 0) & (yte == 1)).sum()
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    prec.append(p); rec.append(r); f1s.append(2 * p * r / (p + r) if p + r else 0)
    alert.append(yh.mean())
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.plot(qs, prec, label="Precision"); ax.plot(qs, rec, label="Recall")
ax.plot(qs, f1s, label="F1-score")
ax.plot(qs, alert, ls=":", c="k", label="Alert rate (fraction flagged)")
ax.set_xlabel("Score threshold (percentile of anomaly score)")
ax.set_ylabel("Value"); ax.set_ylim(0, 1.02)
ax.set_title("Isolation Forest threshold sensitivity (test set, seed 101)")
ax.legend(); ax.grid(alpha=0.3)
save("Fig6_threshold_sensitivity")

# ================ Fig 7: leakage ablation PR overlay ================
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, sc, ls in [("IF, with resp_h (A)", stA, "-"), ("IF, without resp_h (B)", stB, "--"),
                     ("RF, with resp_h (A)", pt_rf, "-"), ("RF, without resp_h (B)", ptB_rf, "--")]:
    p, r, _ = precision_recall_curve(yte, sc)
    ax.plot(r, p, ls, label=f"{name} (AP={average_precision_score(yte, sc):.3f})")
ax.axhline(yte.mean(), ls=":", c="grey", lw=1)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Feature-ablation (target-leakage) analysis, test set")
ax.legend(loc="lower left", fontsize=7.5); ax.grid(alpha=0.3)
save("Fig7_ablation_pr")

# ================ Fig 8: timeout sweep ================
S = pd.read_csv("/home/claude/v2_results/timeout_sweep.csv")
sif = S[S.model == "IF"]
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.plot(sif.timeout_h, sif.f1_mean, "o-", label="F1-score")
ax.plot(sif.timeout_h, sif.ap_mean, "s-", label="Average precision")
ax.plot(sif.timeout_h, sif.auc_roc_mean, "^-", label="AUC-ROC")
ax.plot(sif.timeout_h, sif.prevalence, "d:", c="grey", label="Label prevalence")
ax.set_xscale("log"); ax.set_xticks(sif.timeout_h); ax.set_xticklabels([f"{t:g}" for t in sif.timeout_h])
ax.set_xlabel("Delay-label timeout (hours, log scale)")
ax.set_ylabel("Value"); ax.set_ylim(0, 1.02)
ax.set_title("Sensitivity to delay-label definition (IF, 5 seeds)")
ax.legend(); ax.grid(alpha=0.3)
save("Fig8_timeout_sweep")

# ================ Fig 9: prevalence sensitivity ================
P = pd.read_csv("/home/claude/v2_results/prevalence_sensitivity.csv")
pif = P[P.model == "IF"]
fig, ax = plt.subplots(figsize=(5.8, 3.4))
for m, lab, mk in [("precision", "Precision", "o"), ("recall", "Recall", "s"),
                   ("f1", "F1-score", "^"), ("auc_roc", "AUC-ROC", "d")]:
    ax.errorbar(pif.prevalence_target * 100, pif[f"{m}_mean"],
                yerr=[pif[f"{m}_mean"] - pif[f"{m}_ci_lo"], pif[f"{m}_ci_hi"] - pif[f"{m}_mean"]],
                marker=mk, capsize=3, label=lab)
ax.set_xlabel("Anomaly prevalence (%) — downsampled positives, timeout = 2.0 h")
ax.set_ylabel("Value"); ax.set_ylim(0, 1.02)
ax.set_title("Low-prevalence operating regime (IF, 5 seeds, mean ± 95% CI)")
ax.legend(); ax.grid(alpha=0.3)
save("Fig9_prevalence_sensitivity")

# ================ Supplementary ================
# S1: confusion matrix with Actual/Predicted labels
tval = e2.pick_threshold_val(svA, yva)
cm = confusion_matrix(yte, (stA >= tval).astype(int))
fig, ax = plt.subplots(figsize=(3.4, 3.0))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Anomaly"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
ax.set_title("Isolation Forest confusion matrix\n(test set, seed 101)")
save("FigS1_cm_IF")

# S2: response-time histogram (cleaned, with exclusions annotated)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.hist(dfc["resp_h"], bins=80)
ax.set_yscale("log")
ax.set_xlabel("Response time (hours)"); ax.set_ylabel("Count (log scale)")
ax.set_title("Response-time distribution after cleaning (n = 75,321)")
ax.annotate(f"Zero-duration closures: {audit['n_zero']:,} (retained)\n"
            f"Missing closure: {audit['n_missing_closure']:,} (excluded)\n"
            f"Negative intervals: {audit['n_negative']} (excluded)",
            xy=(0.45, 0.72), xycoords="axes fraction", fontsize=8,
            bbox=dict(fc="white", ec="grey", alpha=0.9))
save("FigS2_hist_response_time")

# S3: category distribution — full-data proportions, honest labeling
vc = dfc["_category"].value_counts()
top = vc.head(8); other = vc.iloc[8:].sum()
lab = list(top.index) + ["Other"]
valpct = list(top / len(dfc) * 100) + [other / len(dfc) * 100]
fig, ax = plt.subplots(figsize=(5.8, 3.0))
ax.bar(lab, valpct, color="#4878b0")
ax.set_ylabel("Share of retained records (%)")
ax.set_title("311 source service categories (agency codes; not clinical categories)")
plt.xticks(rotation=30, ha="right")
for i, v in enumerate(valpct):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.5)
save("FigS3_category_distribution")

# S4-6: ops trade-off (precision vs recall via threshold sweep) per model
for name, sc, fn in [("Isolation Forest", stA, "FigS4_ops_IF"),
                     ("One-Class SVM", st_oc, "FigS5_ops_OCSVM"),
                     ("Random Forest (supervised ref.)", pt_rf, "FigS6_ops_RF")]:
    recs, precs = [], []
    for q in np.linspace(50, 99.5, 80):
        t = np.percentile(sc, q); yh = (sc >= t).astype(int)
        tp = ((yh == 1) & (yte == 1)).sum(); fp = ((yh == 1) & (yte == 0)).sum()
        fn_ = ((yh == 0) & (yte == 1)).sum()
        precs.append(tp / (tp + fp) if tp + fp else 1.0)
        recs.append(tp / (tp + fn_) if tp + fn_ else 0.0)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(recs, precs)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Operational trade-off — {name}\n(threshold percentiles 50–99.5, test set)")
    ax.grid(alpha=0.3)
    save(fn)

# S7: synthetic pipeline — per-type recall (AR=0.10)
SY = pd.read_csv("/home/claude/synth_v2_results/synth_metrics.csv")
sy = SY[SY.anomaly_rate == 0.10].set_index("model")
types = ["delay", "repeat", "volume", "timestamp"]
mods = ["IF", "OCSVM", "RF", "Rule60s"]
labs = ["IF", "OCSVM", "RF (sup.)", "Rule (>60 s)"]
x = np.arange(len(types)); w = 0.2
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for k, (m, lb) in enumerate(zip(mods, labs)):
    vals = [sy.loc[m, f"recall_{t}_mean"] for t in types]
    ax.bar(x + (k - 1.5) * w, vals, w, label=lb)
ax.set_xticks(x); ax.set_xticklabels(["Delayed\nresponse", "Repeated\ncall", "Volume\nburst", "Invalid\ntimestamp"])
ax.set_ylabel("Recall by injected anomaly type"); ax.set_ylim(0, 1.05)
ax.set_title("Synthetic-injection pipeline: per-type recall (AR = 0.10, 10 seeds)")
ax.legend(ncol=4, fontsize=7.5); ax.grid(axis="y", alpha=0.3)
save("FigS7_synth_per_type_recall")

# S8: synthetic pipeline — F1/AP vs injection rate
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for m, lb, mk in [("IF", "IF", "o"), ("OCSVM", "OCSVM", "s"), ("RF", "RF (sup.)", "^")]:
    ss = SY[SY.model == m]
    ax.errorbar(ss.anomaly_rate * 100, ss.f1_mean,
                yerr=[ss.f1_mean - ss.f1_ci_lo, ss.f1_ci_hi - ss.f1_mean],
                marker=mk, capsize=3, label=lb)
ax.set_xlabel("Injection rate (%)  [effective prevalence 15.6 / 27.0 / 44.1%]")
ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.0)
ax.set_title("Synthetic-injection pipeline: F1 vs injection rate (mean ± 95% CI)")
ax.legend(); ax.grid(alpha=0.3)
save("FigS8_synth_f1_vs_rate")

print("figures written:", sorted(os.listdir(FIG)))
