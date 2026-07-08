"""
Churn Signal — Customer Retention Intelligence Console
=======================================================
A production-grade Streamlit application that serves predictions from a
pre-trained scikit-learn pipeline (preprocessing + Gradient Boosting model).

IMPORTANT:
    This app does NOT retrain or rebuild any preprocessing logic. It loads
    the artifact `customer_churn_pipeline.pkl` (a full sklearn Pipeline that
    already includes a ColumnTransformer with StandardScaler + OneHotEncoder,
    followed by a tuned GradientBoostingClassifier) and uses it as-is.

Author: Utkarsh Singh Dahiya
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

MODEL_PATH = "customer_churn_pipeline.pkl"

# Exact column names/order expected by the saved pipeline's ColumnTransformer.
# These come directly from X = df.drop('Churn', axis=1) in the training notebook.
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Risk thresholds used to bucket the predicted churn probability into a
# business-friendly risk tier. Thresholds are expressed as churn probability.
RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}

# Reported offline evaluation metrics (from the training notebook / GridSearchCV).
# Shown in the UI for transparency; NOT recomputed at runtime.
MODEL_METRICS = {
    "Test Accuracy": "79.2%",
    "Precision": "64.7%",
    "Recall": "47.6%",
    "F1 Score": "54.9%",
    "CV ROC AUC": "0.851",
}

# Color tokens for the "signal" visual system, reused across Python-rendered
# HTML/SVG and Plotly figures so everything stays visually consistent.
COLORS = {
    "bg": "#060B14",
    "panel": "#0E1626",
    "panel_border": "#1D2A40",
    "text": "#E7EEF7",
    "muted": "#7C8AA5",
    "accent": "#23D5C4",   # signal teal — calm / low risk
    "warning": "#FFC24E",  # amber — medium risk
    "danger": "#FF5673",   # coral — high risk / churn
}


# =============================================================================
# PAGE CONFIGURATION & GLOBAL STYLES
# =============================================================================

def configure_page() -> None:
    """Set Streamlit page config and inject global CSS styling."""
    st.set_page_config(
        page_title="Churn Signal | Retention Intelligence Console",
        page_icon="📶",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

            :root {
                --bg: #060B14;
                --panel: #0E1626;
                --panel-border: #1D2A40;
                --text: #E7EEF7;
                --muted: #7C8AA5;
                --accent: #23D5C4;
                --warning: #FFC24E;
                --danger: #FF5673;
            }

            /* ---------- Base type ---------- */
            html, body, [class*="css"] {
                font-family: 'IBM Plex Sans', sans-serif;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.5rem;
                max-width: 1180px;
            }

            /* ---------- NOC status strip ---------- */
            .noc-strip {
                display: flex;
                flex-wrap: wrap;
                gap: 22px;
                align-items: center;
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 8px;
                padding: 9px 18px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.74rem;
                letter-spacing: 0.03em;
                color: var(--muted);
                margin-bottom: 1.6rem;
            }
            .noc-strip b { color: var(--text); font-weight: 600; }
            .noc-dot {
                display: inline-block;
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: var(--accent);
                margin-right: 6px;
                box-shadow: 0 0 8px var(--accent);
                animation: dotpulse 2.2s ease-in-out infinite;
            }
            @keyframes dotpulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.35; }
            }

            /* ---------- Header ---------- */
            .app-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--text);
                letter-spacing: -0.01em;
                margin-bottom: 0.15rem;
            }
            .app-title .bars { color: var(--accent); margin-right: 10px; }
            .app-subtitle {
                font-size: 1.0rem;
                color: var(--muted);
                margin-bottom: 1.2rem;
                max-width: 720px;
            }

            /* ---------- Section headers (numbered steps) ---------- */
            .step-eyebrow {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                color: var(--accent);
                margin-bottom: 2px;
            }
            .section-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.15rem;
                font-weight: 600;
                color: var(--text);
                margin: 0 0 0.8rem 0;
            }
            .section-block {
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 12px;
                padding: 20px 22px 8px 22px;
                margin-bottom: 1.1rem;
            }

            /* ---------- Result cards ---------- */
            .result-banner {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.3rem;
                font-weight: 600;
                padding: 14px 18px;
                border-radius: 10px;
                margin-bottom: 1.1rem;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .result-banner.risk-high { background: rgba(255, 86, 115, 0.12); color: var(--danger); border: 1px solid rgba(255,86,115,0.35); }
            .result-banner.risk-medium { background: rgba(255, 194, 78, 0.12); color: var(--warning); border: 1px solid rgba(255,194,78,0.35); }
            .result-banner.risk-low { background: rgba(35, 213, 196, 0.12); color: var(--accent); border: 1px solid rgba(35,213,196,0.35); }

            .risk-badge {
                display: inline-block;
                padding: 4px 13px;
                border-radius: 999px;
                font-family: 'IBM Plex Mono', monospace;
                font-weight: 600;
                font-size: 0.75rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .risk-badge.risk-low { background: rgba(35,213,196,0.15); color: var(--accent); }
            .risk-badge.risk-medium { background: rgba(255,194,78,0.15); color: var(--warning); }
            .risk-badge.risk-high { background: rgba(255,86,115,0.15); color: var(--danger); }

            /* ---------- Signal Pulse Risk Ring (signature element) ---------- */
            .pulse-wrap {
                position: relative;
                width: 210px;
                height: 210px;
                margin: 10px auto 6px auto;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .pulse-ring {
                position: absolute;
                border-radius: 50%;
                border: 1px solid currentColor;
                opacity: 0;
                animation: ringpulse 3s ease-out infinite;
            }
            .pulse-wrap.risk-low .pulse-ring { color: var(--accent); }
            .pulse-wrap.risk-medium .pulse-ring { color: var(--warning); }
            .pulse-wrap.risk-high .pulse-ring { color: var(--danger); }
            .pulse-ring.r1 { width: 100%; height: 100%; animation-delay: 0s; }
            .pulse-ring.r2 { width: 100%; height: 100%; animation-delay: 1s; }
            .pulse-ring.r3 { width: 100%; height: 100%; animation-delay: 2s; }
            @keyframes ringpulse {
                0% { transform: scale(0.72); opacity: 0.55; }
                100% { transform: scale(1.18); opacity: 0; }
            }
            @media (prefers-reduced-motion: reduce) {
                .pulse-ring, .noc-dot { animation: none !important; }
            }
            .progress-ring { transform: rotate(-90deg); width: 172px; height: 172px; }
            .ring-bg { fill: none; stroke: var(--panel-border); stroke-width: 10; }
            .ring-fg { fill: none; stroke-width: 10; stroke-linecap: round; transition: stroke-dashoffset 0.6s ease; }
            .pulse-center {
                position: absolute;
                text-align: center;
            }
            .pulse-value {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.2rem;
                font-weight: 700;
                color: var(--text);
                line-height: 1;
            }
            .pulse-label {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.68rem;
                letter-spacing: 0.08em;
                color: var(--muted);
                text-transform: uppercase;
                margin-top: 4px;
            }

            /* ---------- Confidence split bar ---------- */
            .split-bar-track {
                display: flex;
                width: 100%;
                height: 34px;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid var(--panel-border);
                margin-bottom: 6px;
            }
            .split-seg {
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.75rem;
                font-weight: 600;
                color: #06111f;
            }
            .split-caption {
                display: flex;
                justify-content: space-between;
                font-size: 0.78rem;
                color: var(--muted);
                margin-bottom: 18px;
            }

            /* ---------- Driver chips (model explainability) ---------- */
            .driver-chip {
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 8px;
                padding: 9px 12px;
                margin-bottom: 8px;
            }
            .driver-chip-top {
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                color: var(--text);
                margin-bottom: 5px;
            }
            .driver-chip-top .imp {
                font-family: 'IBM Plex Mono', monospace;
                color: var(--accent);
                font-size: 0.78rem;
            }
            .driver-bar-track {
                width: 100%;
                height: 5px;
                background: var(--panel-border);
                border-radius: 3px;
                overflow: hidden;
            }
            .driver-bar-fill {
                height: 100%;
                background: var(--accent);
                border-radius: 3px;
            }

            /* ---------- Recommendations ---------- */
            .recommend-item {
                background: var(--panel);
                border-left: 3px solid var(--accent);
                border-radius: 6px;
                padding: 10px 14px;
                margin-bottom: 8px;
                font-size: 0.93rem;
                color: var(--text);
            }

            /* ---------- Buttons ---------- */
            .stButton>button {
                width: 100%;
                height: 3em;
                border-radius: 10px;
                font-weight: 600;
                font-size: 1rem;
                background: var(--accent);
                color: #06111f;
                border: none;
                transition: all 0.15s ease-in-out;
            }
            .stButton>button:hover {
                filter: brightness(1.1);
                transform: translateY(-1px);
            }

            /* ---------- Misc ---------- */
            hr { border-color: var(--panel-border); }
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip() -> None:
    """Render a small network-operations-style status strip under the header."""
    st.markdown(
        f"""
        <div class="noc-strip">
            <span><span class="noc-dot"></span><b>MODEL ONLINE</b></span>
            <span>DATASET&nbsp;<b>7,043 RECORDS</b></span>
            <span>ALGORITHM&nbsp;<b>GRADIENT BOOSTING</b></span>
            <span>CV ROC-AUC&nbsp;<b>{MODEL_METRICS['CV ROC AUC']}</b></span>
            <span>TEST ACCURACY&nbsp;<b>{MODEL_METRICS['Test Accuracy']}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading trained model pipeline...")
def load_model(model_path: str):
    """
    Load the pre-trained sklearn Pipeline from disk.

    The pipeline already contains the fitted ColumnTransformer
    (StandardScaler on numeric columns, OneHotEncoder on categorical columns)
    plus the tuned GradientBoostingClassifier. No preprocessing or training
    logic is recreated here — the artifact is used exactly as saved.

    Returns:
        The loaded pipeline object, or None if loading failed.
    """
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(
            f"❌ Model file not found at '{model_path}'. "
            "Please make sure `customer_churn_pipeline.pkl` is in the same "
            "directory as this app."
        )
        return None
    except Exception as exc:  # noqa: BLE001 - surface any load error to the user
        st.error(f"❌ Failed to load model pipeline: {exc}")
        return None


@st.cache_data(show_spinner=False)
def get_global_feature_importance(_model) -> pd.DataFrame:
    """
    Read feature importances directly off the already-fitted
    GradientBoostingClassifier inside the pipeline. Purely introspective —
    does NOT refit or retrain anything.
    """
    pre = _model.named_steps["preprocessor"]
    clf = _model.named_steps["model"]
    names = pre.get_feature_names_out()
    importances = clf.feature_importances_
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def humanize_feature_name(raw_name: str) -> tuple:
    """
    Convert a ColumnTransformer output feature name (e.g.
    'cat__Contract_Month-to-month' or 'num__tenure') into a
    (readable_label, source_column, category_value_or_None) tuple.
    """
    if raw_name.startswith("num__"):
        col = raw_name[len("num__"):]
        return col, col, None

    if raw_name.startswith("cat__"):
        rest = raw_name[len("cat__"):]
        for col in CATEGORICAL_FEATURES:
            prefix = col + "_"
            if rest.startswith(prefix):
                value = rest[len(prefix):]
                return f"{col} = {value}", col, value

    return raw_name, raw_name, None


def compute_key_drivers(model, customer: dict, top_n: int = 6) -> list:
    """
    Cross-reference the model's global feature importances with this
    customer's actual profile to surface which top drivers are 'active'
    for them. This is a lightweight, model-introspective explanation —
    not SHAP, but grounded in the real fitted model rather than
    hand-written rules.
    """
    importance_df = get_global_feature_importance(model)
    drivers = []
    for _, row in importance_df.iterrows():
        label, col, value = humanize_feature_name(row["feature"])
        if value is None:
            # Numeric feature — always relevant, show its actual value.
            drivers.append({
                "label": f"{col} = {customer.get(col)}",
                "importance": row["importance"],
                "active": True,
            })
        else:
            # Categorical dummy — only relevant if the customer matches it.
            is_active = customer.get(col) == value
            drivers.append({
                "label": label,
                "importance": row["importance"],
                "active": is_active,
            })
        if len([d for d in drivers if d["active"]]) >= top_n:
            break

    return [d for d in drivers if d["active"]][:top_n]


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_inputs(tenure: int, monthly_charges: float, total_charges: float) -> list:
    """
    Run business-logic sanity checks on numeric inputs.

    Returns a list of human-readable warning/error messages. An empty list
    means all checks passed.
    """
    issues = []

    if tenure < 0 or tenure > 72:
        issues.append("Tenure must be between 0 and 72 months.")

    if monthly_charges <= 0:
        issues.append("Monthly charges must be greater than 0.")

    if total_charges < 0:
        issues.append("Total charges cannot be negative.")

    # Total charges should realistically be at least the monthly charge for
    # any customer who has been billed at least once.
    if tenure >= 1 and total_charges < monthly_charges * 0.9:
        issues.append(
            "Total charges look inconsistent with tenure and monthly charges "
            "(too low for the given tenure). Please double-check the values."
        )

    return issues


# =============================================================================
# RISK & RECOMMENDATION LOGIC
# =============================================================================

def get_risk_level(probability: float) -> tuple:
    """Map a churn probability to a (label, css_class) risk tier."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk", "risk-low"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"


def generate_recommendations(customer: dict, will_churn: bool, probability: float) -> list:
    """
    Generate business-oriented retention recommendations based on the
    prediction outcome and the customer's profile. Rule-based — presented
    alongside (not instead of) the model-driven driver analysis above.
    """
    recs = []

    if not will_churn and probability < RISK_THRESHOLDS["low"]:
        recs.append("✅ Customer shows strong loyalty signals — maintain standard engagement.")
        recs.append("💡 Consider upselling premium add-ons (streaming, device protection).")
        return recs

    if customer["Contract"] == "Month-to-month":
        recs.append("📄 Offer an incentive to upgrade to a 1- or 2-year contract (e.g., discount or free add-on).")

    if customer["tenure"] < 12:
        recs.append("🤝 Assign a dedicated onboarding/success contact — early-tenure customers are highest risk.")

    if customer["InternetService"] == "Fiber optic":
        recs.append("📶 Review fiber service pricing/quality complaints; fiber customers show elevated churn.")

    if customer["PaymentMethod"] == "Electronic check":
        recs.append("💳 Encourage migration to automatic payment (credit card/bank transfer) with a small incentive.")

    if customer["OnlineSecurity"] == "No" and customer["InternetService"] != "No":
        recs.append("🔐 Offer a free trial of Online Security add-on to increase perceived value.")

    if customer["TechSupport"] == "No" and customer["InternetService"] != "No":
        recs.append("🛠️ Bundle Tech Support at a discount for the next billing cycle.")

    if customer["MonthlyCharges"] > 80:
        recs.append("💰 Review pricing tier — high monthly charges combined with risk signals may warrant a loyalty discount.")

    if not recs:
        recs.append("📞 Schedule a proactive check-in call to understand satisfaction levels.")

    recs.append("📊 Log this customer in the retention campaign tracker for follow-up.")

    return recs


# =============================================================================
# UI SECTIONS — INPUT FORM
# =============================================================================

def render_personal_information() -> dict:
    """Render the Personal Information input section."""
    st.markdown('<div class="step-eyebrow">STEP 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", ["Female", "Male"], help="Customer's gender")
    senior_citizen = c2.selectbox("Senior Citizen", ["No", "Yes"], help="Is the customer 65 or older?")
    partner = c3.selectbox("Has Partner", ["Yes", "No"], help="Does the customer have a partner?")
    dependents = c4.selectbox("Has Dependents", ["Yes", "No"], help="Does the customer have dependents?")

    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
    }


def render_account_information() -> dict:
    """Render the Account Information input section (tenure & contract)."""
    st.markdown('<div class="step-eyebrow">STEP 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📑 Account Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    tenure = c1.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12, step=1,
        help="Number of months the customer has stayed with the company (0–72)."
    )
    contract = c2.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = c3.selectbox("Paperless Billing", ["Yes", "No"])

    return {
        "tenure": int(tenure),
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
    }


def render_internet_services() -> dict:
    """Render the Internet Services input section."""
    st.markdown('<div class="step-eyebrow">STEP 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌐 Internet Services</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    phone_service = c1.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = c3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    c1, c2, c3 = st.columns(3)
    online_security = c1.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = c2.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = c3.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    c1, c2, c3 = st.columns(3)
    tech_support = c1.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = c2.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = c3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    return {
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
    }


def render_billing_information(default_tenure: int) -> dict:
    """Render the Billing Information input section."""
    st.markdown('<div class="step-eyebrow">STEP 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💳 Billing Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    payment_method = c1.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = c2.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0, step=0.5,
        help="Amount charged to the customer monthly."
    )
    total_charges = c3.number_input(
        "Total Charges ($)", min_value=0.0, max_value=20000.0,
        value=round(default_tenure * monthly_charges, 2), step=1.0,
        help="Total amount charged to the customer over their tenure."
    )

    return {
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }


# =============================================================================
# UI SECTIONS — SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    """Render the sidebar with project, dataset, model, and developer info."""
    with st.sidebar:
        st.markdown("## 📶 Churn Signal")
        st.markdown("###### Retention Intelligence Console")
        st.markdown("---")

        st.markdown("#### 📘 About This Project")
        st.markdown(
            "This console predicts whether a telecom customer is likely "
            "to **churn** (cancel their subscription), enabling proactive "
            "retention strategies for the business."
        )

        st.markdown("#### 🗂️ Dataset")
        st.markdown(
            "- **Source:** Telco Customer Churn dataset (IBM sample dataset)\n"
            "- **Records:** 7,043 customers\n"
            "- **Features:** 19 customer attributes\n"
            "- **Target:** Churn (Yes/No)"
        )

        st.markdown("#### 🤖 Model")
        st.markdown(
            "- **Algorithm:** Gradient Boosting Classifier\n"
            "- **Tuning:** GridSearchCV (5-fold CV, ROC AUC scoring)\n"
            "- **Preprocessing:** StandardScaler (numeric) + "
            "OneHotEncoder (categorical), wrapped in a single "
            "scikit-learn Pipeline"
        )

        st.markdown("#### 📊 Reported Performance")
        for metric, value in MODEL_METRICS.items():
            c1, c2 = st.columns([1.4, 1])
            c1.markdown(f"<span style='font-size:0.85rem'>{metric}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='font-weight:700'>{value}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 👨‍💻 Developer")
        st.markdown(
            "**Name:** Utkarsh Singh Dahiya\n\n"
            "**Role:** Data Scientist / ML Engineer"
        )

        st.markdown("---")
        st.caption("Built with Streamlit · scikit-learn · Plotly")


# =============================================================================
# UI SECTIONS — RESULTS
# =============================================================================

def render_signal_pulse_ring(probability: float, risk_class: str) -> None:
    """Render the signature 'signal pulse' circular risk indicator."""
    circumference = 2 * np.pi * 70
    offset = circumference * (1 - probability)
    color_map = {"risk-low": COLORS["accent"], "risk-medium": COLORS["warning"], "risk-high": COLORS["danger"]}
    ring_color = color_map[risk_class]

    st.markdown(
        f"""
        <div class="pulse-wrap {risk_class}">
            <div class="pulse-ring r1"></div>
            <div class="pulse-ring r2"></div>
            <div class="pulse-ring r3"></div>
            <svg class="progress-ring" viewBox="0 0 160 160">
                <circle class="ring-bg" cx="80" cy="80" r="70"></circle>
                <circle class="ring-fg" cx="80" cy="80" r="70"
                    stroke="{ring_color}"
                    stroke-dasharray="{circumference:.1f}"
                    stroke-dashoffset="{offset:.1f}"></circle>
            </svg>
            <div class="pulse-center">
                <div class="pulse-value">{probability:.0%}</div>
                <div class="pulse-label">Churn Probability</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_split(probability: float) -> None:
    """Render a segmented 'signal strength' style bar for stay vs churn odds."""
    stay_pct = (1 - probability) * 100
    churn_pct = probability * 100
    st.markdown(
        f"""
        <div class="split-bar-track">
            <div class="split-seg" style="width:{stay_pct:.1f}%; background:{COLORS['accent']};">
                {stay_pct:.0f}%
            </div>
            <div class="split-seg" style="width:{churn_pct:.1f}%; background:{COLORS['danger']};">
                {churn_pct:.0f}%
            </div>
        </div>
        <div class="split-caption">
            <span>◀ Likely to Stay</span>
            <span>Likely to Churn ▶</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_key_drivers(model, customer: dict) -> None:
    """Render model-introspective 'why this prediction' driver chips."""
    drivers = compute_key_drivers(model, customer, top_n=5)
    if not drivers:
        st.caption("No dominant drivers detected for this profile.")
        return

    max_importance = max(d["importance"] for d in drivers)
    for d in drivers:
        bar_width = (d["importance"] / max_importance) * 100
        st.markdown(
            f"""
            <div class="driver-chip">
                <div class="driver-chip-top">
                    <span>{d['label']}</span>
                    <span class="imp">{d['importance']:.1%}</span>
                </div>
                <div class="driver-bar-track">
                    <div class="driver-bar-fill" style="width:{bar_width:.0f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_result(model, customer: dict, probability: float, will_churn: bool) -> None:
    """Render the full result section: banner, pulse ring, split bar, drivers."""
    risk_label, risk_class = get_risk_level(probability)

    banner_text = "⚠️ Churn Risk Detected" if will_churn else "✅ Customer Likely to Stay"
    st.markdown(
        f"""
        <div class="result-banner {risk_class}">
            {banner_text}
            <span class="risk-badge {risk_class}">{risk_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.3])

    with col1:
        render_signal_pulse_ring(probability, risk_class)

    with col2:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        render_confidence_split(probability)
        st.markdown(
            "<div style='font-family:IBM Plex Mono, monospace; font-size:0.72rem; "
            "letter-spacing:0.08em; color:#7C8AA5; text-transform:uppercase; margin-bottom:8px;'>"
            "📡 Why This Prediction — Active Model Drivers</div>",
            unsafe_allow_html=True,
        )
        render_key_drivers(model, customer)


def render_recommendations(customer: dict, will_churn: bool, probability: float) -> None:
    """Render the business recommendations section."""
    st.markdown('<div class="section-title">💡 Business Recommendations</div>', unsafe_allow_html=True)
    recs = generate_recommendations(customer, will_churn, probability)
    for rec in recs:
        st.markdown(f'<div class="recommend-item">{rec}</div>', unsafe_allow_html=True)


# =============================================================================
# UI SECTIONS — MODEL INSIGHTS TAB
# =============================================================================

def render_model_insights_tab(model) -> None:
    """Render the global model insights tab (feature importance, metrics)."""
    st.markdown('<div class="section-title">📡 Global Feature Importance</div>', unsafe_allow_html=True)
    st.caption(
        "Read directly from the trained Gradient Boosting model's "
        "`feature_importances_` — no retraining performed."
    )

    importance_df = get_global_feature_importance(model).head(12).copy()
    importance_df["label"] = importance_df["feature"].apply(lambda x: humanize_feature_name(x)[0])
    importance_df = importance_df.sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["label"],
            orientation="h",
            marker=dict(
                color=importance_df["importance"],
                colorscale=[[0, COLORS["panel_border"]], [1, COLORS["accent"]]],
            ),
            text=[f"{v:.1%}" for v in importance_df["importance"]],
            textposition="outside",
            textfont=dict(color=COLORS["text"], family="IBM Plex Mono"),
        )
    )
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="IBM Plex Sans"),
        xaxis=dict(showgrid=False, tickformat=".0%", showticklabels=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📊 Reported Model Performance</div>', unsafe_allow_html=True)
    cols = st.columns(len(MODEL_METRICS))
    for col, (metric, value) in zip(cols, MODEL_METRICS.items()):
        col.metric(metric, value)

    st.info(
        "These metrics come from the original training notebook's GridSearchCV "
        "evaluation and are displayed as-is — they are not recomputed here."
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    configure_page()

    st.markdown(
        '<div class="app-title"><span class="bars">📶</span>Churn Signal</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">Real-time churn risk scoring and retention '
        'intelligence for telecom subscribers — powered by a tuned Gradient '
        'Boosting model.</div>',
        unsafe_allow_html=True,
    )
    render_status_strip()

    render_sidebar()

    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()  # Halt execution — cannot proceed without the model artifact

    tab_predict, tab_insights = st.tabs(["🔮  Predict Churn", "📡  Model Insights"])

    with tab_predict:
        with st.form("customer_input_form"):
            personal = render_personal_information()
            account = render_account_information()
            internet = render_internet_services()
            billing = render_billing_information(default_tenure=account["tenure"])

            submitted = st.form_submit_button("🔍 Predict Churn")

        if not submitted:
            st.info("👆 Fill in the customer details above and click **Predict Churn** to see the result.")
        else:
            customer = {**personal, **account, **internet, **billing}

            issues = validate_inputs(
                tenure=customer["tenure"],
                monthly_charges=customer["MonthlyCharges"],
                total_charges=customer["TotalCharges"],
            )
            if issues:
                for issue in issues:
                    st.error(f"⚠️ {issue}")
            else:
                try:
                    input_df = pd.DataFrame([customer])[ALL_FEATURES]
                except KeyError as exc:
                    st.error(f"❌ Column mismatch when building model input: {exc}")
                    input_df = None

                if input_df is not None:
                    try:
                        prediction = model.predict(input_df)[0]
                        probability = model.predict_proba(input_df)[0][1]  # P(Churn = Yes)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"❌ Prediction failed: {exc}")
                        prediction, probability = None, None

                    if prediction is not None:
                        will_churn = bool(prediction == 1)

                        st.markdown("---")
                        render_prediction_result(model, customer, probability, will_churn)

                        st.markdown("---")
                        render_recommendations(customer, will_churn, probability)

                        with st.expander("🔎 View Model Input Data"):
                            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    with tab_insights:
        render_model_insights_tab(model)


if __name__ == "__main__":
    main()
