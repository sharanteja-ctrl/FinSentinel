"""
Fraud Risk Scoring and Categorization Engine for FinSentinel.
Translates model probabilities and anomaly signatures into an intuitive 0-100 score.
"""

from typing import Dict, Tuple


def calculate_risk_score(fraud_probability: float, raw_amount: float = 0.0) -> Tuple[int, str, Dict[str, str]]:
    """
    Computes a calibrated Fraud Risk Score from 0 to 100 based on the model's
    predicted fraud probability and transaction context.
    
    Risk Tiers:
        0 - 30   : LOW RISK    (Genuine / Normal activity)
        31 - 70  : MEDIUM RISK (Suspicious / Elevated review required)
        71 - 100 : HIGH RISK   (Critical / Immediate fraud block / 2FA trigger)
        
    Returns:
        (risk_score, risk_level, metadata)
    """
    # Base risk score directly derived from fraud probability percentage
    # In fraud detection, even a probability of 0.30 is significantly higher than
    # the 0.0017 base prior probability, so non-linear calibration is academically sound.
    prob = max(0.0, min(1.0, float(fraud_probability)))
    
    # Sigmoidal / piece-wise calibration:
    # If prob <= 0.05 -> score scaled linearly 0 to 15
    # If 0.05 < prob <= 0.50 -> score scaled 15 to 65
    # If prob > 0.50 -> score scaled 65 to 100
    if prob <= 0.05:
        score = int((prob / 0.05) * 15)
    elif prob <= 0.50:
        score = int(15 + ((prob - 0.05) / 0.45) * 50)
    else:
        score = int(65 + ((prob - 0.50) / 0.50) * 35)

    # Contextual check: extremely high amounts ($3,000+) on borderline cases push risk up
    if raw_amount > 2500 and score >= 25:
        score = min(100, score + 5)

    score = max(0, min(100, score))

    if score <= 30:
        risk_level = "LOW RISK"
        status_color = "#10B981"  # Emerald green
        action = "Approve Transaction"
        alert_needed = False
    elif score <= 70:
        risk_level = "MEDIUM RISK"
        status_color = "#F59E0B"  # Amber orange
        action = "Trigger 2FA Verification / Analyst Review"
        alert_needed = False
    else:
        risk_level = "HIGH RISK"
        status_color = "#EF4444"  # Crimson red
        action = "Block Transaction & Alert Cardholder"
        alert_needed = True

    metadata = {
        "risk_score": score,
        "risk_level": risk_level,
        "fraud_probability": round(prob * 100, 2),
        "color": status_color,
        "recommended_action": action,
        "alert_needed": alert_needed
    }

    return score, risk_level, metadata
