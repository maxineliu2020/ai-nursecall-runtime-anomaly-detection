import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Nurse-Call Runtime Anomaly Detection (Demo)",
    layout="wide",
)

st.title("AI Nurse-Call Runtime Anomaly Detection – Demo App")
st.write(
    """
This is a **lightweight demo** for the JHU 695.715 Assured Autonomy course.

- If you **upload your own CSV**, the app will use that file.
- If you **don't upload anything**, it will use the built-in small demo dataset
  (`small_demo.csv`) stored in this repo.
"""
)

# 1. 选择数据源：上传 / 内置 small_demo.csv
uploaded_file = st.file_uploader(
    "Upload a nurse-call / service-ticket CSV file (optional)",
    type=["csv"],
    help="If you skip this, the built-in demo file `small_demo.csv` will be used.",
)

if uploaded_file is not None:
    st.success("Using your uploaded CSV file.")
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded – using built-in demo dataset `small_demo.csv`.")
    demo_path = Path(__file__).parent / "small_demo.csv"
    df = pd.read_csv(demo_path)

st.subheader("1. Raw data preview")
st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(20), use_container_width=True)

# 2. 基本统计信息
with st.expander("Show basic statistics", expanded=False):
    st.write("### Summary statistics")
    st.write(df.describe(include="all"))

# 3. 选择用于异常检测的数值特征
st.subheader("2. Simple anomaly detection (Isolation Forest)")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning(
        "No numeric columns found in the dataset. "
        "Please upload a CSV with at least one numeric column "
        "if you want to run anomaly detection."
    )
else:
    default_features = numeric_cols  # 默认使用所有数值列
    selected_features = st.multiselect(
        "Select numeric features for anomaly detection",
        options=numeric_cols,
        default=default_features,
        help="You can deselect columns that are IDs, codes, etc.",
    )

    if selected_features:
        X = df[selected_features].fillna(df[selected_features].median())

        col1, col2 = st.columns(2)
        with col1:
            contamination = st.slider(
                "Estimated fraction of anomalies in the data",
                min_value=0.01,
                max_value=0.30,
                value=0.05,
                step=0.01,
            )
        with col2:
            random_state = st.number_input(
                "Random seed", min_value=0, max_value=9999, value=42, step=1
            )

        if st.button("Run anomaly detection"):
            with st.spinner("Running Isolation Forest..."):
                model = IsolationForest(
                    contamination=contamination,
                    random_state=random_state,
                )
                preds = model.fit_predict(X)

            df_result = df.copy()
            # IsolationForest: -1 = anomaly, 1 = normal
            df_result["anomaly_flag"] = np.where(preds == -1, 1, 0)

            n_anom = int(df_result["anomaly_flag"].sum())
            n_total = df_result.shape[0]

            st.success(
                f"Detected **{n_anom} anomalies** out of **{n_total} records** "
                f"({n_anom / max(n_total,1):.1%})."
            )

            st.write("### Sample of detected anomalies")
            st.dataframe(
                df_result[df_result["anomaly_flag"] == 1].head(50),
                use_container_width=True,
            )

            st.write("### Anomaly flag distribution")
            st.bar_chart(df_result["anomaly_flag"].value_counts())
    else:
        st.info("Please select at least one numeric feature to run anomaly detection.")

st.markdown("---")
st.caption(
    "Demo only – models and thresholds are simplified for visualization purposes. "
    "For full experiments, please refer to the source code in the `src/` folder."
)
