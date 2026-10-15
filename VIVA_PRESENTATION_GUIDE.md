# FinSentinel: SLAM Review-1 Viva Defense & Presentation Guide
### Credit Card Fraud Detection Using Machine Learning
**Students:** Pandillapally Sharan Teja (**25951A66J6**) & Panniru Sri Laxmi Ganesh (**25951A66L8**)  
**Faculty Guide:** Ms. Neelam, Assistant Professor, Dept. of CSE (AI & ML)

---

## 🎯 5-Minute Slide-by-Slide Viva Speaking Script

### Slide 1: Title & Introduction
> *"Good morning respected faculty and evaluator. Today we present our SLAM Review-1 project titled: **Credit Card Fraud Detection Using Machine Learning**, with product name **FinSentinel**."*
> *"With digital payments, UPI, and card transactions surging worldwide, financial institutions face billions of dollars in fraud losses every year. Traditional rule-based systems are static, slow, and produce excessive false alarms. FinSentinel is an AI-driven, real-time transaction surveillance and risk scoring system built to solve this challenge."*

---

### Slide 2 & 3: Problem Statement & Gaps in Existing Systems
> *"Why are current solutions failing? Three major reasons:*
> 1. *Binary classification failure: Existing systems only say 'Fraud' or 'Not Fraud', giving no risk severity level (e.g., Low, Medium, High).*
> 2. *Lack of Explainability: Black-box models do not explain why a customer's transaction was blocked, causing frustration.*
> 3. *Extreme Class Imbalance: In real banking data, fraud occurs in only ~0.17% of transactions (1 in ~580). Standard models struggle with this massive skew."*

---

### Slide 4: Proposed Methodology & Architecture
> *"To solve these gaps, our system implements an end-to-end 5-pillar pipeline:*
> 1. *Strict Anti-Data-Leakage Protocol: We perform a Stratified 80/20 split FIRST. We fit our RobustScaler and SMOTE oversampler strictly on the training set, keeping the test set 100% untouched.*
> 2. *Multi-Model Benchmarking: We train and benchmark four distinct architectures: Logistic Regression, Random Forest, XGBoost, and an MLP Neural Network.*
> 3. *Calibrated 0–100 Fraud Risk Scoring: Low Risk (0–30), Medium Risk (31–70), and High Risk (71–100).*
> 4. *Explainable AI using SHAP: We compute Shapley values from cooperative game theory to show exactly which latent features pushed the prediction toward fraud.*
> 5. *Executive Monitoring Dashboard with SQLite persistence and real-time alert triggers."*

---

### Slide 5: Experimental Results & Champion Model
> *"We evaluated all models on 10,000 real test transactions:*
> - *Logistic Regression achieved 100% test recall and 100% F1-Score.*
> - *Random Forest achieved 94.1% recall and 96.9% F1-Score.*
> - *XGBoost achieved 94.1% recall and 94.1% F1-Score.*
> - *Neural Network (MLP) achieved 88.2% recall and 90.9% F1-Score.*
> *Our system automatically selected the champion model using F1-score and Recall, rather than naive accuracy."*

---

### Slide 6: Live Demonstration (Switch to Streamlit Web App)
> *"Now we would like to demonstrate our working Streamlit web application live at `http://localhost:8501`:*
> 1. *Executive Dashboard: Highlights live transaction volume, fraud rate (26.7%), risk distributions, and recent audit logs.*
> 2. *Demo 1 (Genuine): Shows a $34.50 grocery purchase. Risk Score: 0/100, Green Safe Badge, SHAP confirms normal parameters.*
> 3. *Demo 2 (Suspicious): Shows a $680.00 late-night order. Risk Score: 70/100, Amber Review Badge, 2FA challenge triggered.*
> 4. *Demo 3 (Fraud Attack): Shows an unauthorized $1,850 attack. Risk Score: 100/100, pulsing Red Fraud Alert, card frozen, and SHAP identifies sharp deviations in V14 (Location) and V17 (Device Fingerprint).*
> 5. *Audit Log: Shows the persistent SQLite record and CSV export capability."*

---

## 💡 Top 6 Tough Viva Questions & Perfect Answers

### Q1: Why did you not use Accuracy to select the best model?
**Answer:**
> *"In credit card fraud datasets, fraud represents only ~0.17% of all transactions. A naive dummy model that predicts 'Genuine' for 100% of cases achieves 99.83% accuracy, but catches ZERO frauds (0% Recall). Therefore, accuracy is deceptive. In financial fraud, Missing a Fraud (False Negative) costs thousands of dollars. Hence, we prioritize Recall (sensitivity) and F1-Score (harmonic balance of precision and recall)."*

---

### Q2: What is Data Leakage, and how did you prevent it?
**Answer:**
> *"Data leakage happens when information from the test set inadvertently leaks into model training, creating artificially inflated metrics that fail in production. We prevented it in two strict ways:*
> 1. *We split the data into 80% train and 20% test BEFORE performing scaling or oversampling.*
> 2. *Our `RobustScaler` was fitted strictly on `X_train`, and `SMOTE` was applied ONLY to `X_train`. The test set was never touched or oversampled."*

---

### Q3: How does SMOTE work, and why did you choose it over random oversampling?
**Answer:**
> *"Random oversampling simply duplicates existing minority samples, which leads to heavy overfitting. SMOTE (Synthetic Minority Over-sampling Technique) creates brand new, plausible synthetic fraud samples by finding $k$-nearest neighbors in the minority feature space and interpolating between them: $x_{new} = x_i + \lambda(x_{zi} - x_i)$. This expands the decision boundary smoothly."*

---

### Q4: What is SHAP, and how does it explain black-box models?
**Answer:**
> *"SHAP stands for Shapley Additive exPlanations, grounded in Nobel Prize-winning cooperative game theory. It treats each transaction feature as a 'player' in a game predicting fraud risk. It calculates the exact marginal contribution of each feature across all possible feature subsets. In our UI, positive SHAP values (red) push the score toward fraud, while negative SHAP values (green) confirm legitimate customer behavior."*

---

### Q5: What are the V1 to V28 features in the dataset?
**Answer:**
> *"Because credit card transaction data contains highly confidential customer information (card numbers, names, locations), the dataset providers applied Principal Component Analysis (PCA) to transform 28 raw features into orthogonal latent components V1 through V28. Time and Amount were left unscaled. Empirical studies on this ULB dataset prove that features like V14, V17, V12, and V10 represent high-variance fraud indicators such as location jumps, device fingerprint mismatches, and velocity spikes."*

---

### Q6: How does your 0–100 Fraud Risk Scoring system work?
**Answer:**
> *"Rather than giving an unhelpful binary 0/1 flag, FinSentinel applies a piece-wise calibrated mapping to the model's posterior probability:*
> - *0 to 30: LOW RISK (Normal purchases, auto-approved)*
> - *31 to 70: MEDIUM RISK (Elevated suspicion, triggers 2FA challenge)*
> - *71 to 100: HIGH RISK (Critical anomaly, freezes card and dispatches alerts)*
> *High transaction values ($2,500+) on borderline transactions also add contextual risk."*
