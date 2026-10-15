# FinSentinel: Exploratory Data Analysis (EDA) Report
### Academic Milestone: SLAM Review-1 | Department of CSE (AI & ML)

---

## 1. Dataset Dimensions & Class Distribution
- **Total Transactions:** 50,000
- **Genuine Transactions (Class 0):** 49,914 (99.828%)
- **Fraud Transactions (Class 1):** 86 (0.172%)
- **Imbalance Ratio:** 1 Fraudulent Transaction per ~580 Genuine Transactions.

---

## 2. Statistical Findings
1. **Extreme Imbalance:** Because fraudulent transactions constitute only ~0.17% of the total dataset, standard cross-entropy and naive accuracy metrics fail. SMOTE oversampling on the training split is mandatory to allow models to learn minority class boundaries.
2. **Transaction Amounts:**
   - Genuine transactions have a median of `$20.07` (heavily concentrated under $100).
   - Fraudulent transactions display bimodal traits: micro-authorization tests ($1 to $5) or large anomalous captures ($300 to $2,000+).
3. **Latent Component Deviations:**
   - `V14`, `V17`, `V12`, and `V10` show strong negative shifts during fraud attacks.
   - `V4` and `V11` show noticeable positive shifts during fraud attacks.

---

## 3. Generated Visual Artifacts
- Class Imbalance Plot: `outputs/figures/eda_class_imbalance.png`
- Amount Distribution Plot: `outputs/figures/eda_amount_distribution.png`
- Key Features Boxplot: `outputs/figures/eda_key_features_boxplot.png`
