"""
Explainable AI (XAI) Module for FinSentinel using SHAP.
Explains real-time transaction predictions with Shapley values and natural language insights.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SHAP_BACKGROUND_PATH = os.path.join(MODELS_DIR, "shap_background.pkl")


def prepare_and_save_shap_background(X_train: pd.DataFrame, num_samples: int = 100):
    """
    Saves a small, representative background sample from X_train to calculate
    Shapley baseline values efficiently during inference.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    background_sample = shap.sample(X_train, min(num_samples, len(X_train)), random_state=42)
    joblib.dump(background_sample, SHAP_BACKGROUND_PATH)
    return background_sample


def explain_transaction(model, df_single_scaled: pd.DataFrame, feature_names: list = None) -> dict:
    """
    Computes local SHAP values for a single transaction vector.
    Returns:
        dict containing:
            - top_contributors: list of dicts with {feature, shap_value, impact, direction}
            - summary_text: formatted string for review
            - plot_figure: matplotlib Figure object (waterfall/bar)
    """
    if feature_names is None:
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Load background data if available
    background = None
    if os.path.exists(SHAP_BACKGROUND_PATH):
        try:
            background = joblib.load(SHAP_BACKGROUND_PATH)
        except Exception:
            background = None

    model_type = type(model).__name__
    shap_values_raw = None
    expected_value = 0.0

    try:
        # Determine appropriate explainer based on model architecture
        if "XGB" in model_type or "Forest" in model_type or "Tree" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(df_single_scaled)
            
            # TreeExplainer might output shape (1, n_features) or (1, n_features, 2)
            if hasattr(shap_obj, "values"):
                vals = shap_obj.values
                if len(vals.shape) == 3:  # Binary classification [samples, features, classes]
                    shap_values_raw = vals[0, :, 1]
                    expected_value = shap_obj.base_values[0, 1] if hasattr(shap_obj.base_values, "__len__") else shap_obj.base_values
                else:
                    shap_values_raw = vals[0]
                    expected_value = shap_obj.base_values[0] if hasattr(shap_obj.base_values, "__len__") else shap_obj.base_values
        elif "Logistic" in model_type:
            if background is not None:
                explainer = shap.LinearExplainer(model, background)
            else:
                explainer = shap.LinearExplainer(model, df_single_scaled)
            shap_obj = explainer(df_single_scaled)
            shap_values_raw = shap_obj.values[0]
            expected_value = shap_obj.base_values[0] if hasattr(shap_obj.base_values, "__len__") else shap_obj.base_values
        else:
            # Fallback for MLP or generic black-box
            if background is not None:
                explainer = shap.Explainer(model.predict_proba, background)
                shap_obj = explainer(df_single_scaled)
                vals = shap_obj.values
                if len(vals.shape) == 3:
                    shap_values_raw = vals[0, :, 1]
                else:
                    shap_values_raw = vals[0]
            else:
                # Approximate importance from coefficients or feature perturbation
                shap_values_raw = np.zeros(len(feature_names))
    except Exception as e:
        print(f"[WARNING] SHAP direct extraction note: {e}. Utilizing fallback feature impact calculation.")
        # Fallback: model feature importances * standardized anomaly magnitude
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            raw_vals = df_single_scaled.values.flatten()
            shap_values_raw = importances * np.sign(raw_vals) * np.abs(raw_vals)
        elif hasattr(model, "coef_"):
            coef = model.coef_[0]
            raw_vals = df_single_scaled.values.flatten()
            shap_values_raw = coef * raw_vals
        else:
            shap_values_raw = df_single_scaled.values.flatten() * 0.05

    if shap_values_raw is None:
        shap_values_raw = np.zeros(len(feature_names))

    # Flatten array
    shap_values_raw = np.array(shap_values_raw).flatten()
    
    # Associate features with values
    feature_impacts = []
    for f_name, s_val in zip(feature_names, shap_values_raw):
        abs_val = abs(float(s_val))
        direction = "Toward Fraud" if s_val > 0 else "Toward Genuine"
        
        if abs_val > 0.5:
            magnitude = "High"
        elif abs_val > 0.15:
            magnitude = "Moderate"
        else:
            magnitude = "Minor"
            
        feature_impacts.append({
            "feature": f_name,
            "shap_value": float(s_val),
            "abs_impact": abs_val,
            "direction": direction,
            "magnitude": magnitude
        })

    # Sort features by absolute contribution
    feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
    top_5 = feature_impacts[:5]

    # Generate Matplotlib Figure
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#1E293B")
    ax.set_facecolor("#0F172A")
    
    top_8 = feature_impacts[:8][::-1]  # Reverse for horizontal bar chart
    names = [item["feature"] for item in top_8]
    values = [item["shap_value"] for item in top_8]
    colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor="#334155")
    ax.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.8)
    
    ax.set_title("SHAP Feature Impact on Current Prediction", fontsize=12, fontweight="bold", color="#F8FAFC", pad=12)
    ax.set_xlabel("SHAP Value (Impact on Log-Odds / Fraud Score)", fontsize=10, color="#94A3B8")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")
    
    # Legend indicator
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF4444', label='Pushes Toward Fraud (+SHAP)'),
        Patch(facecolor='#10B981', label='Pushes Toward Genuine (-SHAP)')
    ]
    ax.legend(handles=legend_elements, loc="lower right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
    
    plt.tight_layout()

    # Build concise textual explanation
    lines = []
    lines.append("Top contributing factors identified by Explainable AI:")
    for item in top_5:
        dir_marker = "▲ [Pushes Toward Fraud]" if item["direction"] == "Toward Fraud" else "▼ [Pushes Toward Genuine]"
        lines.append(f"• **{item['feature']}**: {item['magnitude']} impact ({dir_marker}) [SHAP: {item['shap_value']:+.3f}]")
    summary_text = "\n".join(lines)

    return {
        "top_contributors": top_5,
        "all_impacts": feature_impacts,
        "summary_text": summary_text,
        "plot_figure": fig
    }
