[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17767143.svg)](https://doi.org/10.5281/zenodo.17767143)

### 📘 **Runtime Anomaly Detection & Assurance Framework for AI-Driven Nurse Call Systems**

**JHU 695.715 – Assured Autonomy — Course Project**
**Author:** Yuanyuan (Maxine) Liu
**Instructor:** David Concepcion
**Term:** Fall 2025

---

## 🌟 Overview

This repository provides a complete, reproducible anomaly-detection framework for real-world, high-volume service-ticket data.

✨ **It includes:**

* Lightweight anomaly detectors (Isolation Forest, One-Class SVM)
* A supervised assurance baseline (Random Forest)
* An optional TensorFlow autoencoder
* Threshold-sweep tools for safety-critical operations tuning
* SHAP-based interpretability (if installed)
* 18 publication-ready evaluation figures and summary tables
* A Streamlit web demo for interactive exploration

The entire workflow—from raw CSV to figures and metrics—runs in a single script.


---

## 📁 Repository Structure

```
ai-nursecall-runtime-anomaly-detection/
│
├── DATA/                            # Input data (public NYC 311-style CSVs)
│   ├── erm2-nwe9.csv                # Main subset used in the experiments
│   └── 311_ServiceRequest_2010-Present_DataDictionary_Updated_2023.xlsx
│
├── src/                             # Source code
│   ├── experiment_real_plus.py      # Main experiment script (use this one)
│   └── experiment.py                # Older / simplified experiment script
│
├── results/                         # Auto-generated figures & tables
│   ├── ae_train_curve.png
│   ├── alerts_per_hour.png
│   ├── box_delay_by_category.png
│   ├── category_pie.png
│   ├── cm_IF.png
│   ├── heatmap_weekday_hour.png
│   ├── hist_delay.png
│   ├── kde_delay.png
│   ├── metrics_bar_ci.png
│   ├── ops_alerts_vs_th_IF.png
│   ├── ops_alerts_vs_th_OCSVM.png
│   ├── ops_alerts_vs_th_RF.png
│   ├── pr_curves_all.png
│   ├── pr_curves_with_ae.png
│   ├── rf_feature_importance.png
│   ├── shap_bar.png
│   ├── shap_summary.png
│   ├── th_sweep_IF.png
│   ├── real_calls_clean.csv
│   └── summary_metrics.csv
│
├── results000/                      # Earlier experiment run (kept for comparison)
│   ├── metrics_bar_ci_ar010.png
│   └── pr_curves_multi.png
│
├── docs/                            # Paper drafts & written summaries
│   ├── Draft2_Runtime_Anomaly_Detection_and_Assurance_Framework.pdf
│   ├── Experimental Results and Figure Summary.docx
│   └── Read Me.docx
│
├── streamlit/                       # Streamlit demo
│   └── app.py
│
├── requirements.txt                 # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ai-nursecall-runtime-anomaly-detection
cd ai-nursecall-runtime-anomaly-detection
````

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt


```

---

## ▶️ Running the Full Experiment

This single command reproduces all models, figures, metrics, and tables:

```bash
python src/experiment_real_plus.py --data DATA/erm2-nwe9.csv
```

All outputs will appear under:

```
results/
```

This includes:

* PR curves
* ROC-like operational curves
* Confusion matrices
* Threshold sweeps
* Boxplots, KDE, histograms
* Heatmaps
* Feature importances
* SHAP summary & interaction (if SHAP installed)

---

## 🌐 Interactive Streamlit Demo

Launch:

```bash
streamlit run streamlit/app.py
```

Features include:

* Upload service-call CSV
* Autofit ML models
* Interactive PR curves
* Adjustable operational thresholds
* Feature importance visualization
* AE reconstruction error plots
* Real-time anomaly flagging preview

---

## 📊 Key Experimental Results (Core Figures)

The main 8 figures recommended for the paper:

1. **Histogram of Response Time**
2. **KDE of Response Time**
3. **Category Distribution (Top-8)**
4. **Boxplot by Category**
5. **Heatmap (weekday × hour)**
6. **PR Curves (IF / OCSVM / RF)**
7. **PR Curves (AE vs IF)**
8. **RF Feature Importances**

**Primary summary table:** `results/summary_metrics.csv`

All remaining figures are included as appendix materials.

---

## 🔁 Reproducibility

To guarantee reproducibility:

* Same seeds used across models
* Deterministic random splits
* Full pipeline deterministic with the provided CSV
* All plotting code included

https://nursecall-demo.streamlit.app/

```
DOI: to be inserted (Zenodo) https://doi.org/10.5281/zenodo.17767143
```

---

## 🔬 Academic Use & Citation

Please cite this project as:

### **APA**

Liu, Y. (2025). *Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems (Version 1.0)*. GitHub.
[https://github.com/](https://github.com/)maxineliu2020/ai-nursecall-runtime-anomaly-detection

## 📘 How to Cite

If you use this repository, please cite:

Liu, Y. (Maxine). (2025). *Runtime Anomaly Detection and Assurance Framework 
for AI-Driven Nurse Call Systems* (Version 1.0) [Source Code]. Zenodo.  
https://doi.org/10.5281/zenodo.1234567


### **BibTeX**

```bibtex
@software{liu2025nursecall,
  author       = {Yuanyuan (Maxine) Liu},
  title        = {Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems},
  year         = {2025},
  url          = {https://github.com/maxineliu2020/ai-nursecall-runtime-anomaly-detection},
  version      = {1.0},
  note         = {JHU 695.715 Assured Autonomy Course Project}
}
```

---

### 📫 Contact

For questions, collaboration, or citation requests, please contact:

**Yuanyuan (Maxine) Liu**  
Department of Computer Science, 
Johns Hopkins University  
Email:yliu536@jhu.edu | maxineliu2020@gmail.com

---

## 📄 License

This project is released under the **MIT License**, allowing academic and commercial use with attribution.

---

