# 🎨 Streamlit App — AI Nurse-Call Runtime Anomaly Detection & Assurance  
*(Interactive Demo for JHU 695.715 – Assured Autonomy Course Project)*

This folder contains the **interactive Streamlit demo** for the project:

### **Runtime Anomaly Detection and Assurance Framework for AI-Driven Nurse Call Systems**

The app allows users to:

- 📂 Upload & explore nurse-call / service-ticket CSV files  
- ⏱ Compute response-time features (hours, weekday, hour-of-day…)  
- 🤖 Train anomaly detectors  
  - Isolation Forest  
  - One-Class SVM  
  - Random Forest (baseline)  
- 📊 Visualize dataset statistics, model metrics, PR curves, and threshold-sweep results  

This README is intended for **new users, classmates, and reviewers** so that the demo can be reproduced end-to-end smoothly.

---

# 🚀 1. Quick Start Guide (Windows + VS Code)

From the **project root** (where `requirements.txt` and `streamlit/` folder are):

---

## **1. Create a virtual environment**

```bash
python -m venv venv
````

---

## **2. Activate the environment**

### 🔵 PowerShell (recommended)

```powershell
.\venv\Scripts\Activate.ps1
```

If you get an error like *"running scripts is disabled"*, see the troubleshooting section below.

---

### ⬛ Command Prompt (cmd.exe)

```cmd
venv\Scripts\activate.bat
```

---

### 🍎 macOS / Linux

```bash
source venv/bin/activate
```

---

## **3. Install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## **4. Launch the Streamlit App**

```bash
streamlit run streamlit/app.py
```

Then open the URL printed in terminal (usually):

* [http://localhost:8501](http://localhost:8501)
* [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

# 🗂 2. Using Data in the App

The app provides two data-source options in the left sidebar:

---

## **Option A — “Use bundled demo dataset”**

(Works only if a default CSV path is configured in `app.py`.)

---

## **Option B — “Upload your own CSV”**

⭐ **Recommended for reviewers**
⭐ Works perfectly with `DATA/small_demo.csv`

---

## ✨ 2.1 Included Demo Dataset — `DATA/small_demo.csv`

This repository ships with:

```
DATA/small_demo.csv
```

### **What this file is:**

A compact **⚡ 30-row sample** derived from a real-world nurse-call dataset:

* 🧹 Fully **pre-processed**
* 📤 Small enough to upload instantly
* 💻 Lets anyone test the full pipeline without needing a large dataset
* ✔ Contains exactly the columns the app expects

### **Required columns:**

| Column name    | Meaning                          |
| -------------- | -------------------------------- |
| `Created Date` | Timestamp when the ticket opened |
| `Closed Date`  | Timestamp when closed            |
| `category`     | Ticket / nurse-call category     |

### **Optional columns (app will compute if missing):**

| Column       | Description            |
| ------------ | ---------------------- |
| `resp_h`     | Response time in hours |
| `is_anomaly` | Binary anomaly label   |

---

## ✅ How to use `small_demo.csv`

1. Run the app
2. In the sidebar, choose **“Upload your own CSV”**
3. Click **“Browse files”**
4. Select: `DATA/small_demo.csv`
5. Wait for the message:

   > **Uploaded dataset with 30 rows**
6. Now explore the available tabs:

   * 📊 **Dataset overview**
   * 📈 **Model metrics**
   * 📉 **PR curves**
   * 🎚 **Threshold tuning**

This enables classmates and reviewers to fully test the workflow—without needing a large dataset.

---

# 🛠 3. Troubleshooting (Common Issues)

Below are the **exact problems we encountered during development**, with **verified fixes**, so reviewers can reproduce the app without difficulty.

---

## ❗ Issue 3.1 — PowerShell error

### *“The module 'venv' could not be loaded”*

### **Cause:**

PowerShell thinks `venv\Scripts\activate` is a *module*, not a script.

### **Fix:**

Use **relative path with `.\`**

```powershell
.\venv\Scripts\Activate.ps1
```

---

## ❗ Issue 3.2 — PowerShell ExecutionPolicy blocks activation

### Error:

```
File .\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

### **Cause:**

PowerShell prevents execution of local scripts.

### **Fix (choose ONE):**

#### **Option A — Temporary change (recommended)**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

#### **Option B — Permanent change for your user**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Now try:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## ❗ Issue 3.3 — "streamlit : The term 'streamlit' is not recognized"

### **Cause:**

The virtual environment is not activated correctly.

### **Fix:**

```powershell
.\venv\Scripts\Activate.ps1
pip install streamlit
```

---

## ❗ Issue 3.4 — “KeyError: 'Created Date'”

### **Cause:**

Your CSV is missing required columns.

### **Fix:**

Ensure your custom dataset includes:

* `Created Date`
* `Closed Date`
* `category`

Or simply use:

```
DATA/small_demo.csv
```

---

# 🎁 4. Notes for Reviewers & Classmates

* The Streamlit app is fully self-contained.
* `small_demo.csv` guarantees that *all* features (PR curves, threshold tuning, metrics, plots) work even on very small datasets.
* Real-world datasets can be extremely large; this demo ensures smooth testing.

---

# 🙌 Questions / Issues?

Feel free to open a GitHub Issue on the project repository.

Enjoy exploring the model! 🚀

```
