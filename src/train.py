"""
End-to-End Model Training and Benchmarking Pipeline for FinSentinel.
Trains Logistic Regression, Random Forest, XGBoost, and MLP Neural Network.
Applies SMOTE strictly on training split to eliminate data leakage.
Selects best model based on F1-Score and PR-AUC.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Internal module imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.download_data import ensure_dataset
from src.preprocessing import load_and_validate_data, prepare_training_data
from src.evaluate import (
    evaluate_single_model,
    plot_confusion_matrices,
    plot_roc_and_pr_curves,
    plot_metrics_comparison_bar
)
from src.explainability import prepare_and_save_shap_background

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")


def run_training_pipeline(csv_path: str = None, sample_size: int = None):
    """
    Executes the entire model training, benchmarking, and selection pipeline.
    """
    start_time = time.time()
    print("=" * 65)
    print("  FINSENTINEL: REAL-TIME CREDIT CARD FRAUD DETECTION PIPELINE  ")
    print("=" * 65)

    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    
    csv_path = ensure_dataset()

    print(f"\n[STEP 1/6] Ingesting and validating transaction dataset from: {csv_path}")
    df = load_and_validate_data(csv_path, max_samples=sample_size)
    total_samples = len(df)
    fraud_count = int((df["Class"] == 1).sum())
    genuine_count = int((df["Class"] == 0).sum())
    print(f"  • Total samples:   {total_samples:,}")
    print(f"  • Genuine (Class 0): {genuine_count:,} ({genuine_count/total_samples*100:.2f}%)")
    print(f"  • Fraud (Class 1):   {fraud_count:,} ({fraud_count/total_samples*100:.4f}%)")

    print("\n[STEP 2/6] Splitting data & applying anti-leakage SMOTE on training partition only...")
    # 80/20 Stratified Split
    X_train_res, X_test_scaled, y_train_res, y_test, scaler, feature_names = prepare_training_data(
        df,
        test_size=0.20,
        random_state=42,
        apply_smote=True,
        smote_sampling_strategy=0.25 # Oversample minority to 25% of majority for optimal gradient stability
    )
    print(f"  • Training set (after SMOTE): {X_train_res.shape[0]:,} samples (Fraud: {(y_train_res == 1).sum():,})")
    print(f"  • Test set (untouched real):   {X_test_scaled.shape[0]:,} samples (Fraud: {(y_test == 1).sum():,})")

    # Save a background sample for SHAP Explainable AI
    print("  • Saving SHAP background reference sample...")
    prepare_and_save_shap_background(X_train_res, num_samples=100)

    print("\n[STEP 3/6] Defining candidate machine learning models...")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            random_state=42
        )
    }

    print("\n[STEP 4/6] Training and evaluating candidate models...")
    benchmark_results = []
    models_probs = {}
    trained_models = {}

    for name, clf in models.items():
        m_start = time.time()
        print(f"\n  ▶ Training {name}...")
        clf.fit(X_train_res, y_train_res)
        m_duration = time.time() - m_start
        print(f"    ✓ Trained in {m_duration:.2f}s")
        
        # Evaluate on untouched real test set
        res, y_prob = evaluate_single_model(clf, X_test_scaled, y_test, model_name=name)
        res["training_time_seconds"] = round(m_duration, 2)
        benchmark_results.append(res)
        models_probs[name] = y_prob
        trained_models[name] = clf

        print(f"    • Accuracy:  {res['accuracy']*100:.2f}%")
        print(f"    • Precision: {res['precision']*100:.2f}%")
        print(f"    • Recall:    {res['recall']*100:.2f}%")
        print(f"    • F1-Score:  {res['f1_score']*100:.2f}%")
        print(f"    • ROC-AUC:   {res['roc_auc']:.4f}")
        print(f"    • PR-AUC:    {res['pr_auc']:.4f}")

    print("\n[STEP 5/6] Generating evaluation figures & comparison charts...")
    plot_confusion_matrices(benchmark_results)
    plot_roc_and_pr_curves(models_probs, y_test)
    plot_metrics_comparison_bar(benchmark_results)

    print("\n[STEP 6/6] Selecting best model based on F1-Score and Recall...")
    # In fraud detection, missing a fraud (False Negative) is catastrophic.
    # Therefore, we score by weighted combination of F1-score (harmonic mean of precision & recall)
    # and PR-AUC.
    def model_fitness(r):
        return (0.6 * r["f1_score"]) + (0.3 * r["recall"]) + (0.1 * r["roc_auc"])

    best_result = max(benchmark_results, key=model_fitness)
    best_name = best_result["model_name"]
    best_clf = trained_models[best_name]

    print(f"\n{'*'*65}")
    print(f"  BEST PERFORMING MODEL: {best_name.upper()}")
    print(f"  • F1-Score: {best_result['f1_score']*100:.2f}%")
    print(f"  • Recall:   {best_result['recall']*100:.2f}%")
    print(f"  • ROC-AUC:  {best_result['roc_auc']:.4f}")
    print(f"  • Precision:{best_result['precision']*100:.2f}%")
    print(f"{'*'*65}")

    # Serialize best model and metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_clf, best_model_path)
    print(f"  ✓ Saved best model to: {best_model_path}")

    # Also save each individual trained model for interactive comparison in Streamlit
    for name, clf in trained_models.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(clf, os.path.join(MODELS_DIR, f"{slug}.pkl"))

    # Metadata JSON
    metadata = {
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_rows": total_samples,
        "test_set_rows": len(y_test),
        "test_fraud_count": int((y_test == 1).sum()),
        "test_genuine_count": int((y_test == 0).sum()),
        "best_model_name": best_name,
        "best_model_metrics": best_result,
        "feature_names": feature_names,
        "pipeline_version": "1.0.0",
        "imbalance_technique": "SMOTE (training only) + RobustScaler",
        "why_accuracy_misleading": (
            "With an extreme fraud imbalance (0.17%), a trivial model predicting 'genuine' for every single "
            "transaction achieves 99.83% accuracy, but 0.00% recall. Therefore, F1-Score, Recall, and PR-AUC "
            "are the primary scientific benchmarks for fraud detection."
        )
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  ✓ Saved model metadata to: {metadata_path}")

    # All models summary JSON
    summary_path = os.path.join(MODELS_DIR, "all_models_summary.json")
    with open(summary_path, "w") as f:
        # Strip out y_pred to keep file compact
        clean_summary = [{k: v for k, v in r.items() if k != "y_pred"} for r in benchmark_results]
        json.dump(clean_summary, f, indent=4)
    print(f"  ✓ Saved multi-model benchmark summary to: {summary_path}")

    total_time = time.time() - start_time
    print(f"\n[DONE] Complete ML Pipeline finished successfully in {total_time:.2f} seconds!")
    print("=" * 65)

    return metadata, clean_summary


if __name__ == "__main__":
    run_training_pipeline()
