import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    PrecisionRecallDisplay,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import shap


# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="AI Nurse-Call Runtime Anomaly Detection – Demo App",
    layout="wide",
)


# -------------------------------
# Helper functions
# -------------------------------

def load_dataset(uploaded_file):
    """
    Load CSV from upload or from built-in small_demo.csv.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "uploaded"
    else:
        demo_path = Path(__file__).parent / "small_demo.csv"
        df = pd.read_csv(demo_path)
        source = "demo"
    return df, source


def guess_datetime_columns(df: pd.DataFrame):
    """
    Try to guess 'created' and 'closed' datetime columns based on column names.
    """
    created_col = None
    closed_col = None
    for col in df.columns:
        lc = col.lower()
        if created_col is None and any(
            kw in lc for kw in ["create", "start", "open", "request"]
        ):
            created_col = col
        if closed_col is None and any(
            kw in lc for kw in ["close", "end", "resolve", "response", "finish"]
        ):
            closed_col = col

    return created_col, closed_col


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-related features for anomaly detection:
    - resp_minutes: response time in minutes
    - hour: hour of day
    - weekday: 0-6
    - is_weekend: 0/1
    """
    df = df.copy()

    # Try to reuse existing response-time column if available
    resp_cols = [c for c in df.columns if "resp_" in c.lower()]
    if resp_cols:
        resp_col = resp_cols[0]
        try:
            resp_minutes = pd.to_numeric(df[resp_col], errors="coerce") * 60.0
            df["resp_minutes"] = resp_minutes
        except Exception:
            pass

    # If we still don't have resp_minutes, try to derive from datetime cols
    if "resp_minutes" not in df.columns:
        created_col, closed_col = guess_datetime_columns(df)
        if created_col is not None:
            created_dt = pd.to_datetime(
                df[created_col], errors="coerce", utc=False
            )
            df["created_dt"] = created_dt
        else:
            created_dt = None

        if closed_col is not None:
            closed_dt = pd.to_datetime(
                df[closed_col], errors="coerce", utc=False
            )
            df["closed_dt"] = closed_dt
        else:
            closed_dt = None

        if created_dt is not None and closed_dt is not None:
            resp_minutes = (closed_dt - created_dt).dt.total_seconds() / 60.0
            df["resp_minutes"] = resp_minutes

    # Derive hour / weekday features if we have a created_dt
    if "created_dt" in df.columns:
        df["hour"] = df["created_dt"].dt.hour
        df["weekday"] = df["created_dt"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    return df


def select_candidate_numeric_features(df: pd.DataFrame):
    """
    Automatically select reasonable numeric features for anomaly detection.

    Heuristics:
    - numeric dtype
    - exclude obvious ID / geo / zip columns
    - exclude columns with almost-unique values (very likely IDs)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []

    ignore_name_tokens = [
        "id",
        "key",
        "uid",
        "zip",
        "lat",
        "lon",
        "long",
        "x_coord",
        "y_coord",
        "x_coordinate",
        "y_coordinate",
        "phone",
        "number",
    ]

    candidate_cols = []
    n = len(df)

    for col in numeric_cols:
        lc = col.lower()
        if any(tok in lc for tok in ignore_name_tokens):
            continue

        nunique = df[col].nunique(dropna=True)
        # Exclude columns that are almost all unique (likely IDs)
        if n > 0 and nunique > 0.9 * n:
            continue

        candidate_cols.append(col)

    return candidate_cols


def find_binary_label_column(df: pd.DataFrame):
    """
    Try to find a binary ground-truth label column, e.g., is_anomaly.
    """
    # Strong candidates by name
    name_candidates = [
        "is_anomaly",
        "label",
        "y",
        "anomaly",
        "is_outlier",
        "ground_truth",
        "gt",
    ]
    for c in df.columns:
        if c.lower() in name_candidates:
            values = df[c].dropna().unique()
            if set(values).issubset({0, 1}):
                return c

    # Fallback: any column with only 0/1
    for c in df.columns:
        values = df[c].dropna().unique()
        if len(values) and set(values).issubset({0, 1}):
            return c

    return None


def get_reasonable_category_column(df: pd.DataFrame):
    """
    Get a categorical column suitable for grouping anomaly rate:
    e.g., category / complaint_type / dept / ward.
    """
    max_unique = 30  # avoid too-many-category columns

    candidates = []
    for c in df.columns:
        if df[c].dtype == "object":
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= max_unique:
                candidates.append(c)

    priority_order = [
        "category",
        "complaint_type",
        "type",
        "agency_name",
        "department",
        "unit",
        "ward",
        "address_type",
    ]
    for name in priority_order:
        for c in candidates:
            if name in c.lower():
                return c

    return candidates[0] if candidates else None


def train_isolation_forest(
    df_features: pd.DataFrame,
    contamination: float,
    random_state: int,
    max_train_size: int = 20000,
):
    """
    Train Isolation Forest on numeric features.
    Uses subsampling for large datasets to keep training fast.
    Returns model and anomaly scores for all rows.
    """
    X = df_features.values

    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    if n > max_train_size:
        sample_idx = rng.choice(n, size=max_train_size, replace=False)
        X_train = X[sample_idx]
    else:
        X_train = X

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train)

    # decision_function: higher = more normal, so we negate it
    scores = -model.decision_function(X)
    return model, scores


def add_anomaly_flag(
    df: pd.DataFrame, scores: np.ndarray, contamination: float
) -> pd.DataFrame:
    """
    Add 'anomaly_score' and 'anomaly_flag' to df.
    """
    df = df.copy()
    df["anomaly_score"] = scores

    # threshold by quantile
    thresh = np.quantile(scores, 1.0 - contamination)
    df["anomaly_flag"] = (scores >= thresh).astype(int)
    return df


# -------------------------------
# UI Layout
# -------------------------------

st.title("AI Nurse-Call Runtime Anomaly Detection – Demo App")

st.write(
    """
This is a **lightweight but feature-rich demo** for the **JHU 695.715 Assured Autonomy** course.

- If you **upload your own CSV**, the app will use that file.
- If you **don't upload anything**, it will use the built-in small demo dataset
  (`small_demo.csv`) stored in this repo.
"""
)

uploaded_file = st.file_uploader(
    "Upload a nurse-call / service-ticket CSV file (optional)",
    type=["csv"],
)

df_raw, source = load_dataset(uploaded_file)

if source == "uploaded":
    st.success("Using your **uploaded CSV** file.")
else:
    st.info("No file uploaded – using built-in demo dataset `small_demo.csv`.")

# Engineer features
df = engineer_time_features(df_raw)

st.markdown("## 1. Raw data preview")

st.write(f"Shape: **{df_raw.shape[0]} rows × {df_raw.shape[1]} columns**")
st.dataframe(df_raw.head(50), use_container_width=True)

with st.expander("Show basic statistics", expanded=False):
    st.write("### Summary statistics")
    st.write(df_raw.describe(include="all"))


# -------------------------------
# 2. Anomaly detection controls
# -------------------------------
st.markdown("## 2. Simple anomaly detection (Isolation Forest)")

candidate_features = select_candidate_numeric_features(df)

if not candidate_features:
    st.warning(
        "No suitable numeric columns were detected for anomaly detection. "
        "Please make sure your dataset contains numeric features "
        "such as response time, counts, or other metrics."
    )
else:
    st.caption(
        "The app automatically excludes obvious ID/geo columns and almost-unique "
        "numeric columns (likely IDs). You can further adjust the selection below."
    )

    selected_features = st.multiselect(
        "Select numeric features for anomaly detection",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=candidate_features,
        help="You can deselect columns that are IDs, codes, or coordinates.",
    )

    col_c, col_s, col_e = st.columns(3)
    with col_c:
        contamination = st.slider(
            "Estimated fraction of anomalies in the data",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
        )
    with col_s:
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
        )
    with col_e:
        use_shap = st.checkbox(
            "SHAP explainability",
            value=False,
            help=(
                "Compute global feature importance using SHAP (TreeExplainer). "
                "For large datasets a random subset will be used."
            ),
        )

    run_pressed = st.button("Run anomaly detection")

    if run_pressed and selected_features:
        with st.spinner("Running Isolation Forest and computing anomaly scores..."):
            X = df[selected_features].copy()
            X = X.fillna(X.median(numeric_only=True))

            model, scores = train_isolation_forest(
                X, contamination=contamination, random_state=random_state
            )
            df_result = add_anomaly_flag(df, scores, contamination)

        n_anom = int(df_result["anomaly_flag"].sum())
        n_total = df_result.shape[0]

        st.success(
            f"Detected **{n_anom} anomalies** out of **{n_total} records** "
            f"({n_anom / max(n_total, 1):.1%})."
        )

        st.write("### Sample of detected anomalies")
        st.dataframe(
            df_result[df_result["anomaly_flag"] == 1].head(50),
            use_container_width=True,
        )

        st.write("### Anomaly flag distribution")
        st.bar_chart(df_result["anomaly_flag"].value_counts())

        # ------------------------------------------
        # 3.1 SHAP global feature importance (optional)
        # ------------------------------------------
        if use_shap:
            st.subheader("3.1 SHAP-based global feature importance (experimental)")
            with st.spinner("Computing SHAP values for a random subset..."):
                try:
                    max_shap_samples = 2000
                    n = X.shape[0]
                    rng = np.random.default_rng(random_state)
                    if n > max_shap_samples:
                        idx = rng.choice(n, size=max_shap_samples, replace=False)
                        X_shap = X.iloc[idx]
                    else:
                        X_shap = X

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    mean_abs = np.mean(np.abs(shap_values), axis=0)
                    importance_df = pd.DataFrame(
                        {"feature": X_shap.columns, "mean_abs_shap": mean_abs}
                    ).sort_values("mean_abs_shap", ascending=True)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(importance_df["feature"], importance_df["mean_abs_shap"])
                    ax.set_xlabel("Mean |SHAP value| (global importance)")
                    ax.set_ylabel("Feature")
                    ax.grid(True, axis="x", alpha=0.2)
                    st.pyplot(fig)

                    st.caption(
                        "Higher mean |SHAP| indicates features that contribute more "
                        "to pushing a call towards *anomalous* behaviour under the "
                        "current Isolation Forest model."
                    )
                except Exception as e:
                    st.warning(f"SHAP explanation failed: {e}")

        # -------------------------------
        # 3. Advanced visualizations
        # -------------------------------
        st.markdown("## 3. Advanced visualizations")

        # Response time distribution (if available)
        if "resp_minutes" in df_result.columns:
            st.subheader("3.2 Response time distribution")

            fig, ax = plt.subplots(figsize=(6, 4))
            normal = df_result[df_result["anomaly_flag"] == 0]["resp_minutes"]
            anom = df_result[df_result["anomaly_flag"] == 1]["resp_minutes"]

            ax.hist(normal, bins=50, alpha=0.7, label="Normal")
            ax.hist(anom, bins=50, alpha=0.7, label="Anomaly")
            ax.set_xlabel("Response time (minutes)")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.2)

            st.pyplot(fig)

        # Hour-of-day x weekday heatmap
        if {"hour", "weekday"}.issubset(df_result.columns):
            st.subheader("3.3 Call volume heatmap (hour × weekday)")
            pivot = (
                df_result.pivot_table(
                    index="weekday",
                    columns="hour",
                    values="anomaly_flag",
                    aggfunc="count",
                )
                .fillna(0)
                .astype(int)
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Weekday (0=Mon)")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticklabels(pivot.index)
            fig.colorbar(im, ax=ax, label="Call count")

            st.pyplot(fig)

        # Category-wise anomaly rate
        cat_col = get_reasonable_category_column(df_result)
        if cat_col is not None:
            st.subheader(f"3.4 Anomaly rate by category: `{cat_col}`")

            group = df_result.groupby(cat_col)["anomaly_flag"].agg(
                ["mean", "count"]
            )
            group = group.rename(
                columns={"mean": "anomaly_rate", "count": "num_records"}
            )
            st.dataframe(group.sort_values("anomaly_rate", ascending=False))

            st.bar_chart(group["anomaly_rate"])

        # -------------------------------
        # 4. Evaluation (if ground-truth labels exist)
        # -------------------------------
        st.markdown("## 4. Evaluation against ground-truth labels (if available)")
        label_col = find_binary_label_column(df_result)

        if label_col is None:
            st.info(
                "No binary ground-truth label column was detected "
                "(e.g., `is_anomaly`). If you add one with values 0/1, "
                "the app will automatically compute precision/recall/F1 and "
                "a PR curve."
            )
        else:
            st.success(
                f"Detected ground-truth label column: **`{label_col}`** "
                "(values 0/1)."
            )
            y_true = df_result[label_col].astype(int).values
            y_pred = df_result["anomaly_flag"].astype(int).values

            # Basic metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            st.write(
                f"- **Precision**: {precision:.3f}\n"
                f"- **Recall**: {recall:.3f}\n"
                f"- **F1-score**: {f1:.3f}"
            )

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["True Normal (0)", "True Anomaly (1)"],
                columns=["Pred Normal (0)", "Pred Anomaly (1)"],
            )
            st.write("### Confusion matrix")
            st.dataframe(cm_df)

            # PR curve using continuous scores
            if "anomaly_score" in df_result.columns:
                scores = df_result["anomaly_score"].values
                ap = average_precision_score(y_true, scores)

                st.write(f"**Average Precision (AP)**: {ap:.3f}")

                fig, ax = plt.subplots(figsize=(5, 4))
                PrecisionRecallDisplay.from_predictions(
                    y_true, scores, ax=ax
                )
                ax.set_title("Precision–Recall curve")
                st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Demo only – models and thresholds are simplified for visualization purposes. "
    "For full experiments and training pipelines, please refer to the source code "
    "in the `src/` folder."
)
