"""
Real-Time Transaction Inference and Risk Prediction Service for FinSentinel.
Integrates preprocessing, trained ML model scoring, risk level mapping, SHAP explainability, and database logging.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import preprocess_single_transaction, SCALER_PATH
from src.risk_scoring import calculate_risk_score
from src.explainability import explain_transaction
from database.db import insert_transaction

MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")


def load_inference_artifacts():
    """Loads the trained model and fitted scaler for real-time inference."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {BEST_MODEL_PATH}. Please run src/train.py first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Fitted scaler not found at {SCALER_PATH}. Please run src/train.py first.")

    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_transaction(
    features_dict: dict,
    model=None,
    scaler=None,
    save_to_db: bool = True,
    generate_shap: bool = True
) -> dict:
    """
    Analyzes an incoming credit card transaction in real time.
    
    Args:
        features_dict: dictionary containing 'Time', 'Amount', and 'V1' through 'V28'.
        model: optional preloaded ML model.
        scaler: optional preloaded RobustScaler.
        save_to_db: boolean flag to persist record to SQLite.
        generate_shap: boolean flag to compute SHAP explanations.
        
    Returns:
        Structured prediction result dictionary.
    """
    if model is None or scaler is None:
        model, scaler = load_inference_artifacts()

    # Preprocess transaction with exact scaler fitted during training
    scaled_array, df_scaled = preprocess_single_transaction(features_dict, scaler=scaler)

    # Predict class label and probability using DataFrame to preserve feature names
    pred_raw = int(model.predict(df_scaled)[0])
    
    if hasattr(model, "predict_proba"):
        prob_fraud = float(model.predict_proba(df_scaled)[0, 1])
    elif hasattr(model, "decision_function"):
        df_val = float(model.decision_function(df_scaled)[0])
        prob_fraud = 1.0 / (1.0 + np.exp(-df_val)) # Sigmoid
    else:
        prob_fraud = float(pred_raw)

    raw_amount = float(features_dict.get("Amount", 0.0))
    risk_score, risk_level, score_meta = calculate_risk_score(prob_fraud, raw_amount=raw_amount)

    # Label assignment: If risk score > 70 or model predicts 1, mark as FRAUD
    is_fraud = (pred_raw == 1) or (risk_score >= 71)
    prediction_label = "FRAUD" if is_fraud else "GENUINE"

    # Explainable AI via SHAP
    shap_results = None
    top_features_summary = "N/A"
    if generate_shap:
        try:
            shap_results = explain_transaction(model, df_scaled)
            top_factors = [f"{c['feature']} ({c['magnitude']})" for c in shap_results["top_contributors"][:3]]
            top_features_summary = ", ".join(top_factors)
        except Exception as e:
            print(f"[WARNING] SHAP generation skipped: {e}")

    txn_id = f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(100, 999)}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "txn_id": txn_id,
        "timestamp": timestamp,
        "amount": raw_amount,
        "prediction": prediction_label,
        "is_fraud": is_fraud,
        "fraud_probability": round(prob_fraud * 100, 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "status_color": score_meta["color"],
        "recommended_action": score_meta["recommended_action"],
        "alert_needed": score_meta["alert_needed"],
        "top_features": top_features_summary,
        "shap_explanation": shap_results,
        "raw_features": features_dict
    }

    # Persist to SQLite
    if save_to_db:
        try:
            insert_transaction({
                "txn_id": txn_id,
                "timestamp": timestamp,
                "amount": raw_amount,
                "prediction": prediction_label,
                "fraud_probability": result["fraud_probability"],
                "risk_score": risk_score,
                "risk_level": risk_level,
                "top_features": top_features_summary,
                "status": "BLOCKED" if is_fraud else "APPROVED",
                "is_demo": 0
            })
        except Exception as e:
            print(f"[DB ERROR] Could not save transaction: {e}")

    return result
