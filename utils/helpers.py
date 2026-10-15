"""
Faculty Demo Presets, Input Validation, and Formatting Utilities for FinSentinel.
Pre-packages reproducible test vectors for seamless 5-minute college review demonstrations.
"""

import json
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_SAMPLES_PATH = os.path.join(DATA_DIR, "sample_test_transactions.json")

# Faculty Demo Scenarios
DEMO_PRESETS = {
    "Demo 1: Genuine Grocery / Retail Transaction": {
        "description": "Routine supermarket purchase during regular daylight hours. PCA features are all within normal ±1.0 bounds.",
        "expected_result": "GENUINE (Low Risk ~ 0-15/100)",
        "features": {
            "Time": 42500.0,
            "Amount": 34.50,
            "V1": -0.23, "V2": 0.15, "V3": 0.85, "V4": -0.42, "V5": 0.18,
            "V6": -0.12, "V7": 0.31, "V8": 0.05, "V9": 0.22, "V10": -0.15,
            "V11": -0.35, "V12": 0.28, "V13": -0.05, "V14": 0.12, "V15": 0.45,
            "V16": -0.22, "V17": 0.08, "V18": 0.14, "V19": -0.08, "V20": 0.02,
            "V21": -0.08, "V22": -0.15, "V23": 0.04, "V24": 0.12, "V25": -0.18,
            "V26": 0.06, "V27": 0.02, "V28": 0.01
        }
    },
    "Demo 2: Suspicious Borderline Transaction": {
        "description": "Unusual late-night elevated electronics order with moderate latent deviations on fraud-sensitive components (V14, V10).",
        "expected_result": "GENUINE / REVIEW (Medium Risk ~ 35-65/100)",
        "features": {
            "Time": 86400.0,
            "Amount": 680.00,
            "V1": -1.25, "V2": 0.85, "V3": -0.95, "V4": 1.45, "V5": -0.65,
            "V6": -0.45, "V7": -0.85, "V8": 0.45, "V9": -0.85, "V10": -1.85,
            "V11": 1.25, "V12": -1.45, "V13": -0.25, "V14": -2.15, "V15": 0.15,
            "V16": -1.10, "V17": -1.80, "V18": -0.65, "V19": 0.75, "V20": 0.35,
            "V21": 0.25, "V22": 0.15, "V23": -0.12, "V24": -0.25, "V25": 0.22,
            "V26": 0.18, "V27": 0.12, "V28": -0.04
        }
    },
    "Demo 3: High-Risk Coordinated Fraud Attack": {
        "description": "Severe anomaly signature matching compromised card / identity cloning. Extreme negative shifts in V14, V17, V12, V10 and high positive shift in V4.",
        "expected_result": "FRAUDULENT (High Risk ~ 85-100/100) -> Triggers Immediate Alert",
        "features": {
            "Time": 74500.0,
            "Amount": 1850.00,
            "V1": -4.85, "V2": 3.75, "V3": -6.25, "V4": 4.65, "V5": -3.85,
            "V6": -1.85, "V7": -5.95, "V8": 2.85, "V9": -3.45, "V10": -4.85,
            "V11": 3.95, "V12": -6.15, "V13": 0.45, "V14": -7.25, "V15": -0.55,
            "V16": -4.85, "V17": -8.15, "V18": -2.95, "V19": 1.85, "V20": 0.85,
            "V21": 1.15, "V22": 0.25, "V23": -0.45, "V24": -0.35, "V25": 0.55,
            "V26": 0.35, "V27": 0.85, "V28": 0.45
        }
    }
}


def save_demo_presets_json():
    """Serializes demo vectors to data/sample_test_transactions.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TEST_SAMPLES_PATH, "w") as f:
        json.dump(DEMO_PRESETS, f, indent=4)


def load_demo_presets() -> dict:
    """Loads demo presets from JSON or fallback dictionary."""
    if os.path.exists(TEST_SAMPLES_PATH):
        try:
            with open(TEST_SAMPLES_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    save_demo_presets_json()
    return DEMO_PRESETS


def format_currency(amount: float) -> str:
    """Formats numeric amount as USD currency string."""
    return f"${amount:,.2f}"


def format_risk_badge_html(risk_level: str, risk_score: int) -> str:
    """Generates styled HTML badge for risk tiers."""
    colors = {
        "LOW RISK": ("#065F46", "#34D399", "✓"),
        "MEDIUM RISK": ("#78350F", "#FBBF24", "⚠"),
        "HIGH RISK": ("#7F1D1D", "#F87171", "🚨")
    }
    bg, fg, icon = colors.get(risk_level, ("#1E293B", "#94A3B8", "•"))
    return f"""
    <span style="background-color: {bg}; color: {fg}; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; border: 1px solid {fg};">
        {icon} {risk_level} ({risk_score}/100)
    </span>
    """


if __name__ == "__main__":
    save_demo_presets_json()
    print(f"[PRESETS] Saved faculty demo presets to: {TEST_SAMPLES_PATH}")
