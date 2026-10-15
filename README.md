# FinSentinel: AI-Driven Real-Time Credit Card Fraud Detection
### B.Tech 2nd Year CSE (Artificial Intelligence & Machine Learning)
### 3rd-Semester SLAM Project Review-1

---

## 👥 Project Credentials
- **Student 1:** Pandillapally Sharan Teja (Roll No: **25951A66J6**)
- **Student 2:** Panniru Sri Laxmi Ganesh (Roll No: **25951A66L8**)
- **Faculty Guide:** Ms. Neelam, Assistant Professor
- **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)
- **Academic Year:** 2025–2026

---

## 📌 1. Project Overview & Problem Statement
Digital financial transactions and online card payments have grown exponentially. Traditional fraud mitigation approaches rely heavily on static, hard-coded rule engines that:
1. Struggle to detect emerging, evolving fraud vectors.
2. Produce high false-positive rates that inconvenience legitimate cardholders.
3. Fail to provide granular risk severity (binary Fraud / Genuine output only).
4. Lack transparency and explainability regarding why a specific transaction was flagged.

**FinSentinel** is an intelligent, real-time credit card fraud detection system that bridges these gaps by combining:
- **Zero-Leakage Machine Learning Pipeline** with SMOTE imbalance handling.
- **Multi-Model Benchmarking** (Logistic Regression, Random Forest, XGBoost, Neural Networks).
- **Calibrated 0–100 Fraud Risk Scoring Engine** (Low, Medium, High Risk).
- **Explainable AI (XAI) with SHAP** (Shapley Additive exPlanations) for transparent model decision audits.
- **Real-Time Security Dashboard & Alert System** with persistent SQLite transaction logging.

---

## 🏗️ 2. System Architecture & ML Pipeline

```
Raw Transaction Dataset (creditcard.csv)
       │
       ▼
Data Ingestion & Integrity Check (30 features: Time, V1-V28, Amount, Class)
       │
       ▼
Stratified Train/Test Split (80% Train, 20% Test) [ANTI-LEAKAGE BOUNDARY]
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
RobustScaler (Fitted on Train ONLY)       Transform Test Set
       │                                          │
       ▼                                          │
SMOTE (Oversampling on Train ONLY)                │
       │                                          │
       ▼                                          │
Model Training Suite                              │
├── Logistic Regression                           │
├── Random Forest Classifier                      │
├── XGBoost Gradient Boosting                     │
└── Multi-Layer Perceptron (MLP Neural Net)       │
       │                                          │
       ▼                                          ▼
Model Evaluation & Benchmark (Precision, Recall, F1, ROC-AUC, PR-AUC)
       │
       ▼
Automated Selection of Best Model (F1 & Recall Weighted)
       │
       ▼
Model Serialization (Joblib: best_model.pkl, scaler.pkl, metadata.json)
       │
       ▼
FinSentinel Streamlit Web Platform (app.py)
 ├── 🏠 Executive Dashboard (Real-time KPIs & Risk Distributions)
 ├── 🔍 Real-Time Fraud Detection (Interactive Input & Fast Presets)
 ├── 📊 Model Performance Center (ROC, PR Curves, Confusion Matrices)
 ├── 🚨 Fraud Alerts Center (High-risk transaction mitigation)
 ├── 📋 Transaction History (Searchable SQLite audit log with CSV export)
 └── ℹ️ About Project & SLAM Viva Defense Guide
```

---

## 🧠 3. Critical Academic Concepts for Viva Review

### A. Why is Accuracy Misleading for Fraud Detection?
Credit card fraud datasets have severe class imbalance (~0.17% fraud transactions). 
If a naive model predicts **"Genuine" (Class 0)** for 100% of transactions:
$$\text{Accuracy} = \frac{99,830}{100,000} = 99.83\%$$
Despite high accuracy, the model caught **0 frauds** ($\text{Recall} = 0.00\%$). 
Therefore, FinSentinel prioritizes **Recall** (catching every fraud), **Precision** (avoiding false alarms), **F1-Score**, and **PR-AUC**.

### B. Anti-Data-Leakage Protocol
1. **Train/Test Splitting Occurs First:** We never fit scalers or oversamplers on the complete dataset.
2. **Scaler Isolation:** `RobustScaler` (which uses the median and Interquartile Range to prevent transaction amount outliers from distorting weights) is fitted **strictly** on `X_train`.
3. **SMOTE Isolation:** Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic fraud samples **strictly** within `X_train`. The test set remains completely pristine.

### C. Explainable AI (SHAP)
FinSentinel uses Shapley Additive exPlanations grounded in cooperative game theory:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
Where $\phi_i$ represents the exact marginal contribution of feature $i$ toward pushing the prediction toward Fraud (+SHAP) or Genuine (-SHAP).

### D. Fraud Risk Score Calibration (0–100)
- **0 – 30:** **LOW RISK** (Standard legitimate purchase, auto-approved).
- **31 – 70:** **MEDIUM RISK** (Elevated anomaly pattern, requires 2FA or analyst review).
- **71 – 100:** **HIGH RISK** (Critical fraud alert, card authorization blocked).

---

## ⚡ 4. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Terminal (macOS / Linux / Windows PowerShell)

### Step 1: Open the Project Directory
```bash
cd /Users/sharanteja/.gemini/antigravity-ide/scratch/FinSentinel
```

### Step 2: (Optional) Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Machine Learning Training Pipeline
```bash
python src/train.py
```
*This command loads the dataset, applies anti-leakage SMOTE, trains all 4 models (Logistic Regression, Random Forest, XGBoost, and MLP Neural Network), computes all evaluation metrics, selects the best model, and generates visualization charts in `outputs/figures/`.*

### Step 5: Start the Streamlit Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🎯 5. Step-by-Step 5-Minute Faculty Review Demo Script

During your SLAM review, follow this exact sequence:

1. **DEMO 1: Open Dashboard (Page 1)**
   - Show the KPI cards: Total Transactions, Fraud Flagged, Genuine Passed, Fraud Rate.
   - Point out the Risk Tier distribution and live transaction feed.

2. **DEMO 2: Showcase Model Benchmarking (Page 3)**
   - Show the champion model callout.
   - Walk through the comparison table (Logistic Regression, Random Forest, XGBoost, Neural Net).
   - Show the Confusion Matrices and ROC/PR Curves.
   - Explain why Accuracy is deceptive on imbalanced data.

3. **DEMO 3: Test a Genuine Transaction (Page 2)**
   - Click **"Demo 1: Genuine Grocery / Retail Transaction"** preset.
   - Click **"⚡ ANALYZE TRANSACTION WITH FINSENTINEL"**.
   - Show the green banner: `✓ TRANSACTION VERIFIED AS GENUINE`, Risk Score (~0-10/100), and green SHAP features.

4. **DEMO 4: Test a Suspicious Transaction (Page 2)**
   - Select **"Demo 2: Suspicious Borderline Transaction"** preset.
   - Click **"⚡ ANALYZE TRANSACTION"**.
   - Show the orange banner: `⚠️ SUSPICIOUS PATTERN IDENTIFIED`, Medium Risk score (~50-65/100), and 2FA trigger suggestion.

5. **DEMO 5: Test a High-Risk Fraud Transaction (Page 2)**
   - Select **"Demo 3: High-Risk Coordinated Fraud Attack"** preset.
   - Click **"⚡ ANALYZE TRANSACTION"**.
   - Show the pulsing red alert: `🚨 HIGH RISK FRAUD ALERT`, Risk Score (95-100/100), and the SHAP waterfall showing extreme shifts in V14, V17, and V12.

6. **DEMO 6: Check the Fraud Alerts Center (Page 4)**
   - Navigate to **"🚨 Fraud Alerts"**. Show the active incident card and demonstrate the "Freeze Card" simulation button.

7. **DEMO 7: Audit in Transaction History (Page 5)**
   - Show the persistent SQLite log. Filter by Risk Level or Prediction, and show the CSV download feature.

8. **DEMO 8: Summarize Viva Alignment (Page 6)**
   - Conclude on **"ℹ️ About Project"**, pointing out the student roll numbers, faculty guide credit, and theoretical justifications.

---

## 📁 6. Repository File Hierarchy

```
FinSentinel/
├── app.py                             # Multi-page interactive Streamlit web application
├── requirements.txt                   # Tested package requirements
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Standard git ignore patterns
│
├── data/
│   ├── creditcard.csv                 # Authenticated 31-feature fraud dataset
│   └── sample_test_transactions.json  # Curated test vectors for faculty review
│
├── models/
│   ├── best_model.pkl                 # Serialized champion model
│   ├── scaler.pkl                     # Fitted RobustScaler object
│   ├── model_metadata.json            # Model performance records & parameters
│   ├── all_models_summary.json        # Full 4-model comparison benchmark
│   ├── shap_background.pkl           # Saved baseline sample for SHAP explanations
│   ├── logistic_regression.pkl        # Serialized candidate model
│   ├── random_forest.pkl              # Serialized candidate model
│   ├── xgboost.pkl                    # Serialized candidate model
│   └── neural_network_mlp.pkl         # Serialized candidate model
│
├── src/
│   ├── __init__.py                    # Source module initializer
│   ├── download_data.py               # Dataset ingestion & download helper
│   ├── preprocessing.py               # Anti-leakage split, scaling, and SMOTE
│   ├── train.py                       # Training pipeline & model selection
│   ├── evaluate.py                    # Confusion matrices, ROC & PR curves
│   ├── predict.py                     # Real-time transaction inference engine
│   ├── risk_scoring.py                # 0-100 Risk score mathematical mapping
│   └── explainability.py              # SHAP Shapley value extraction & waterfall charts
│
├── database/
│   ├── __init__.py
│   ├── db.py                          # SQLite database engine & queries
│   └── transactions.db                # Persistent database file
│
├── outputs/
│   └── figures/
│       ├── confusion_matrices.png     # Heatmaps for all 4 models
│       ├── roc_pr_curves.png          # Combined ROC and PR benchmark curves
│       └── models_comparison_bar.png  # Multi-metric grouped comparison chart
│
└── utils/
    ├── __init__.py
    ├── helpers.py                     # Faculty demo presets & formatters
    └── visualizer.py                  # Streamlit CSS, risk gauge & metric cards
```

---

## ⚖️ 7. Academic Integrity & Ethics Statement
This project is an **academic prototype** developed for collegiate laboratory demonstration. It does not interface with production payment processing switches or transmit live cardholder credentials. All evaluations, confusion matrices, and metrics presented reflect real algorithmic execution on empirical datasets.
