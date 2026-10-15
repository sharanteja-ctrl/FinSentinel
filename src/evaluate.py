"""
Model Evaluation and Performance Metrics Module for FinSentinel.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC and produces publication-ready visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")


def evaluate_single_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluates a single classification model on untouched test data.
    Computes comprehensive imbalanced classification metrics.
    """
    y_pred = model.predict(X_test)
    
    # Check if model provides probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        # Normalize decision function into 0-1
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    else:
        y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "y_pred": y_pred.tolist() if len(y_pred) < 10000 else None,
        "classification_report_str": classification_report(y_test, y_pred, digits=4, zero_division=0)
    }
    
    return results, y_prob


def plot_confusion_matrices(models_results: list, save_path: str = None):
    """Generates comparison confusion matrix subplots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")

    n_models = len(models_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), facecolor="#0F172A")
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, models_results):
        ax.set_facecolor("#1E293B")
        cm = res["confusion_matrix"]
        cm_matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Genuine (0)", "Fraud (1)"],
            yticklabels=["Genuine (0)", "Fraud (1)"],
            annot_kws={"size": 11, "weight": "bold", "color": "#0F172A"}
        )
        ax.set_title(f"{res['model_name']}\nF1: {res['f1_score']:.3f} | Recall: {res['recall']:.3f}", color="#F8FAFC", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", color="#94A3B8", fontsize=9)
        ax.set_ylabel("True Label", color="#94A3B8", fontsize=9)
        ax.tick_params(colors="#94A3B8")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return save_path


def plot_roc_and_pr_curves(models_probs: dict, y_test, save_path: str = None):
    """
    Plots multi-model ROC Curves and Precision-Recall Curves side-by-side.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "roc_pr_curves.png")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0F172A")
    ax_roc.set_facecolor("#1E293B")
    ax_pr.set_facecolor("#1E293B")

    colors = ["#38BDF8", "#34D399", "#F472B6", "#FBBF24"]

    for idx, (name, y_prob) in enumerate(models_probs.items()):
        color = colors[idx % len(colors)]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

        # PR Curve
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax_pr.plot(rec_c, prec_c, color=color, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], color="#64748B", linestyle="--", lw=1)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", color="#F8FAFC", fontsize=11, fontweight="bold", pad=10)
    ax_roc.set_xlabel("False Positive Rate", color="#94A3B8")
    ax_roc.set_ylabel("True Positive Rate (Recall)", color="#94A3B8")
    ax_roc.tick_params(colors="#94A3B8")
    ax_roc.grid(True, linestyle=":", alpha=0.2, color="#475569")
    ax_roc.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)

    # PR formatting
    baseline_fraud_rate = float((y_test == 1).sum() / len(y_test))
    ax_pr.axhline(baseline_fraud_rate, color="#64748B", linestyle="--", lw=1, label=f"No-Skill Baseline ({baseline_fraud_rate:.4f})")
    ax_pr.set_title("Precision-Recall Curve (Imbalance Benchmark)", color="#F8FAFC", fontsize=11, fontweight="bold", pad=10)
    ax_pr.set_xlabel("Recall", color="#94A3B8")
    ax_pr.set_ylabel("Precision", color="#94A3B8")
    ax_pr.tick_params(colors="#94A3B8")
    ax_pr.grid(True, linestyle=":", alpha=0.2, color="#475569")
    ax_pr.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return save_path


def plot_metrics_comparison_bar(models_results: list, save_path: str = None):
    """
    Plots a grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC across models.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "models_comparison_bar.png")

    df_res = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1_score"],
            "ROC-AUC": r["roc_auc"]
        }
        for r in models_results
    ])

    df_melted = df_res.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")

    palette = ["#38BDF8", "#34D399", "#FBBF24", "#F472B6", "#A78BFA"]
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=palette, ax=ax)

    ax.set_title("Multi-Model Comparative Performance Benchmark", color="#F8FAFC", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 to 1.0)", color="#94A3B8")
    ax.set_xlabel("Machine Learning Algorithm", color="#94A3B8")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#94A3B8")
    ax.grid(axis="y", linestyle=":", alpha=0.2, color="#475569")
    ax.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return save_path
