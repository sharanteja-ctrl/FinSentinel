"""
Visual Components, Themes, and Dashboard Styling for FinSentinel.
Provides custom CSS, glassmorphism cards, risk meters, and executive KPI badges.
"""

import streamlit as st

# Human-understandable mappings for PCA features to help faculty & reviewers
FEATURE_DESCRIPTIONS = {
    "V14": "Geographic / IP Location Discrepancy (Distance from usual home location)",
    "V17": "Device & Browser Fingerprint Variance (Unrecognized device / VPN usage)",
    "V12": "Merchant Terminal Authenticity & Industry Risk Index",
    "V10": "Billing vs. Shipping Address Match Confidence",
    "V4": "Transaction Velocity Spike (Multiple attempts in short duration)",
    "V11": "Card Testing Latent Signature (Pattern of small auth probes)",
    "V16": "Payment Channel & Gateway Security Score",
    "V18": "Behavioral Biometric Profile Deviation",
    "Amount": "Transaction Dollar Value (Compared to 30-day historical mean)",
    "Time": "Time of Transaction (Hour-of-day sleep cycle anomaly)"
}


def get_feature_plain_english(feature_name: str) -> str:
    """Returns an intuitive, plain-English translation for technical features."""
    return FEATURE_DESCRIPTIONS.get(feature_name, f"Latent Transaction Characteristic ({feature_name})")


def apply_custom_css():
    """Injects high-fidelity fintech theme CSS into Streamlit."""
    custom_css = """
    <style>
        /* Modern Fintech Dark Styling with Outfit / Plus Jakarta Sans */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stMarkdown, .stText {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.02em;
        }

        /* Glassmorphism Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 16px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent);
            opacity: 0.7;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 14px 36px 0 rgba(14, 165, 233, 0.15);
        }

        .metric-title {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .metric-value {
            color: #F8FAFC;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.15;
            font-family: 'Outfit', sans-serif;
        }

        .metric-sub {
            color: #64748B;
            font-size: 0.8rem;
            margin-top: 8px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        /* Result Callout Banners with Glowing Borders */
        .alert-fraud {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.9) 0%, rgba(69, 10, 10, 0.98) 100%);
            border: 2px solid #EF4444;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
            border-radius: 16px;
            padding: 24px;
            color: #FEE2E2;
            margin-bottom: 22px;
            animation: pulse-border 2.5s infinite;
        }

        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
            70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .alert-genuine {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.9) 0%, rgba(2, 44, 34, 0.98) 100%);
            border: 2px solid #10B981;
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
            border-radius: 16px;
            padding: 24px;
            color: #D1FAE5;
            margin-bottom: 22px;
        }

        .alert-review {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.9) 0%, rgba(69, 26, 3, 0.98) 100%);
            border: 2px solid #F59E0B;
            box-shadow: 0 0 25px rgba(245, 158, 11, 0.25);
            border-radius: 16px;
            padding: 24px;
            color: #FEF3C7;
            margin-bottom: 22px;
        }

        /* Speedometer / Meter Gauge Container */
        .gauge-wrapper {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 22px;
            margin: 14px 0;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }

        .risk-bar-container {
            width: 100%;
            height: 22px;
            background: #090D16;
            border-radius: 9999px;
            overflow: hidden;
            border: 1px solid #334155;
            margin: 14px 0 8px 0;
            position: relative;
        }

        .risk-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        /* Interactive Step Box */
        .step-box {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            height: 100%;
            transition: all 0.2s ease;
        }

        .step-box:hover {
            border-color: #38BDF8;
            background: rgba(30, 41, 59, 0.85);
            transform: translateY(-2px);
        }

        .step-number {
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            background: #0284C7;
            color: white;
            border-radius: 50%;
            font-weight: 800;
            font-size: 0.85rem;
            margin-bottom: 8px;
        }

        /* Streamlit Button Customization */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.02em !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6) !important;
        }

        /* Preset Fast-Button Styles */
        .preset-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.82rem;
            margin-right: 8px;
            margin-bottom: 8px;
            cursor: pointer;
        }

        /* Modern Sidebar Navigation Links (Eliminate raw radio dots) */
        section[data-testid="stSidebar"] {
            background-color: #0B0F19 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"],
        section[data-testid="stSidebar"] [data-testid="stRadio"] {
            gap: 8px !important;
            display: flex !important;
            flex-direction: column !important;
            padding-top: 4px !important;
        }

        /* Eliminate all default radio buttons, circles, bullets, svgs, and markers */
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:not([data-testid="stMarkdownContainer"]):not(:last-child),
        section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label svg,
        section[data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] > div:first-child,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child,
        section[data-testid="stSidebar"] div[role="radiogroup"] label::before,
        section[data-testid="stSidebar"] div[role="radiogroup"] label::after {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            opacity: 0 !important;
            position: absolute !important;
            pointer-events: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Style each nav item as a sleek modern SaaS pill button */
        section[data-testid="stSidebar"] div[role="radiogroup"] label,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: rgba(30, 41, 59, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            box-sizing: border-box !important;
            position: relative !important;
            overflow: hidden !important;
        }

        /* Modern subtle hover state */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(56, 189, 248, 0.12) !important;
            border-color: rgba(56, 189, 248, 0.35) !important;
            transform: translateX(4px) !important;
            box-shadow: 0 4px 16px rgba(56, 189, 248, 0.15) !important;
        }

        /* Typography for nav items */
        section[data-testid="stSidebar"] div[role="radiogroup"] label p,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
        section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important;
            margin: 0 !important;
            letter-spacing: 0.01em !important;
            transition: color 0.2s ease !important;
            white-space: nowrap !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {
            color: #F8FAFC !important;
        }

        /* Active / Selected Nav State */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has([aria-checked="true"]),
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.4) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
            border: 1px solid #38BDF8 !important;
            box-shadow: 0 4px 20px rgba(2, 132, 199, 0.35) !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has([aria-checked="true"]) p,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }

        /* Glowing indicator bar on the left edge of active item */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before,
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]::before,
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has([aria-checked="true"])::before,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::before {
            content: "" !important;
            display: block !important;
            position: absolute !important;
            left: 0 !important;
            top: 0 !important;
            bottom: 0 !important;
            width: 4px !important;
            height: 100% !important;
            background: #38BDF8 !important;
            box-shadow: 0 0 12px #38BDF8 !important;
            border-radius: 4px 0 0 4px !important;
        }

        /* Hide Streamlit default branding watermark */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """
    st.html(custom_css)


import textwrap


def render_metric_card(title: str, value: str, subtext: str = "", border_color: str = "#38BDF8", icon: str = "📊"):
    """Renders a responsive, glassmorphic metric card widget."""
    card_html = textwrap.dedent(f"""
    <div class="metric-card" style="border-left: 5px solid {border_color};">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """).strip()
    st.html(card_html)


def render_risk_gauge(score: int, level: str, fraud_prob: float):
    """Renders an intuitive, high-impact visual fraud risk speedometer / gauge without markdown escaping."""
    if score <= 30:
        fill_color = "#10B981"  # Emerald
        glow_color = "rgba(16, 185, 129, 0.5)"
        status_text = "SAFE / NORMAL"
        badge_bg = "rgba(16, 185, 129, 0.2)"
        badge_border = "#10B981"
    elif score <= 70:
        fill_color = "#F59E0B"  # Amber
        glow_color = "rgba(245, 158, 11, 0.5)"
        status_text = "ELEVATED SUSPICION"
        badge_bg = "rgba(245, 158, 11, 0.2)"
        badge_border = "#F59E0B"
    else:
        fill_color = "#EF4444"  # Crimson
        glow_color = "rgba(239, 68, 68, 0.6)"
        status_text = "CRITICAL FRAUD THREAT"
        badge_bg = "rgba(239, 68, 68, 0.2)"
        badge_border = "#EF4444"

    gauge_html = textwrap.dedent(f"""
    <div class="gauge-wrapper">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em;">
                    FRAUD RISK ASSESSMENT METER
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {fill_color}; margin-top: 2px;">
                    {level} — {status_text}
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {fill_color}; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 1.3rem;">
                    {score} <span style="font-size: 0.85rem; font-weight: 600; color: #94A3B8;">/ 100</span>
                </span>
            </div>
        </div>
        <div class="risk-bar-container">
            <div class="risk-bar-fill" style="width: {max(6, score)}%; background: linear-gradient(90deg, #10B981, {fill_color}); box-shadow: 0 0 16px {glow_color};"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600; color: #64748B; margin-top: 4px;">
            <span style="color: #10B981;">● 0–30: Low Risk (Auto-Approved)</span>
            <span style="color: #F59E0B;">● 31–70: Medium Risk (Requires 2FA)</span>
            <span style="color: #EF4444;">● 71–100: High Risk (Fraud Block)</span>
        </div>
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #334155; display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8;">
            <span><b>Statistical Model Probability:</b> <span style="color: #F8FAFC; font-weight: 700;">{fraud_prob:.1f}%</span></span>
            <span><b>Recommended System Action:</b> <span style="color: {fill_color}; font-weight: 700;">{'BLOCK TRANSACTION' if score >= 71 else ('TRIGGER 2FA CHALLENGE' if score >= 31 else 'APPROVE TRANSACTION')}</span></span>
        </div>
    </div>
    """).strip()
    st.html(gauge_html)


def render_how_it_works():
    """Renders an intuitive 4-step infographic explaining the system to non-technical users."""
    html = textwrap.dedent("""
    <div style="margin: 20px 0 26px 0;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;">
            💡 HOW FINSENTINEL PROTECTS EVERY TRANSACTION (4-STEP AI PIPELINE):
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
            <div class="step-box">
                <div class="step-number">1</div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">Customer Swipes Card</div>
                <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">Amount, timestamp, merchant type, and device telemetry are ingested instantly.</div>
            </div>
            <div class="step-box">
                <div class="step-number">2</div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">AI Analyzes 30 Signals</div>
                <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">Trained ML ensemble detects subtle deviations in location, frequency, and spending habits.</div>
            </div>
            <div class="step-box">
                <div class="step-number">3</div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">0–100 Risk Score & SHAP</div>
                <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">Generates a transparent risk score with exact human-readable reasons why it was flagged.</div>
            </div>
            <div class="step-box">
                <div class="step-number">4</div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">Instant Safeguard</div>
                <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">Genuine charges settle in <50ms; suspicious charges trigger 2FA; fraudulent cards are frozen.</div>
            </div>
        </div>
    </div>
    """).strip()
    st.html(html)
