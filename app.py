"""
Customer Churn Prediction Dashboard
====================================
A production-grade Streamlit application that serves predictions from a
pre-trained scikit-learn pipeline (preprocessing + Gradient Boosting model).

IMPORTANT:
    This app does NOT retrain or rebuild any preprocessing logic. It loads
    the artifact `customer_churn_pipeline.pkl` (a full sklearn Pipeline that
    already includes a ColumnTransformer with StandardScaler + OneHotEncoder,
    followed by a tuned GradientBoostingClassifier) and uses it as-is.

Author: (Your Name)
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
# Shown in the sidebar for transparency; NOT recomputed at runtime.
MODEL_METRICS = {
    "Test Accuracy": "79.2%",
    "Precision": "64.7%",
    "Recall": "47.6%",
    "F1 Score": "54.9%",
    "CV ROC AUC": "0.851",
}


# =============================================================================
# PAGE CONFIGURATION & GLOBAL STYLES
# =============================================================================

def configure_page() -> None:
    """Set Streamlit page config and inject global CSS styling."""
    st.set_page_config(
        page_title="Customer Churn Prediction | Business Dashboard",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            /* ---------- Global type & layout ---------- */
            html, body, [class*="css"] {
                font-family: 'Segoe UI', 'Inter', sans-serif;
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }

            /* ---------- Header ---------- */
            .app-title {
                font-size: 2.3rem;
                font-weight: 800;
                color: #1a2b48;
                margin-bottom: 0.1rem;
            }
            .app-subtitle {
                font-size: 1.02rem;
                color: #6b7280;
                margin-bottom: 1.4rem;
            }

            /* ---------- Section headers ---------- */
            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #1a2b48;
                border-left: 4px solid #2563eb;
                padding-left: 10px;
                margin: 1.1rem 0 0.7rem 0;
            }

            /* ---------- Cards ---------- */
            .info-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 18px 20px;
                box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
                margin-bottom: 1rem;
            }

            .result-card {
                border-radius: 16px;
                padding: 26px 24px;
                color: #ffffff;
                text-align: center;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.12);
            }
            .result-card.churn {
                background: linear-gradient(135deg, #dc2626 0%, #7f1d1d 100%);
            }
            .result-card.retain {
                background: linear-gradient(135deg, #16a34a 0%, #14532d 100%);
            }
            .result-label {
                font-size: 1.1rem;
                font-weight: 600;
                opacity: 0.9;
            }
            .result-value {
                font-size: 2.4rem;
                font-weight: 800;
                margin: 4px 0 2px 0;
            }

            .risk-badge {
                display: inline-block;
                padding: 5px 14px;
                border-radius: 999px;
                font-weight: 700;
                font-size: 0.85rem;
                letter-spacing: 0.02em;
            }
            .risk-low { background: #dcfce7; color: #166534; }
            .risk-medium { background: #fef9c3; color: #854d0e; }
            .risk-high { background: #fee2e2; color: #991b1b; }

            .recommend-item {
                background: #f8fafc;
                border-left: 3px solid #2563eb;
                border-radius: 6px;
                padding: 10px 14px;
                margin-bottom: 8px;
                font-size: 0.93rem;
                color: #1f2937;
            }

            /* ---------- Buttons ---------- */
            .stButton>button {
                width: 100%;
                height: 3em;
                border-radius: 10px;
                font-weight: 700;
                font-size: 1rem;
                background: #1a2b48;
                color: white;
                border: none;
                transition: all 0.15s ease-in-out;
            }
            .stButton>button:hover {
                background: #2563eb;
                transform: translateY(-1px);
            }

            /* ---------- Sidebar ---------- */
            section[data-testid="stSidebar"] {
                background: #0f172a;
            }
            section[data-testid="stSidebar"] * {
                color: #e2e8f0 !important;
            }
            section[data-testid="stSidebar"] hr {
                border-color: #334155;
            }

            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
        </style>
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
    """
    Map a churn probability to a (label, css_class) risk tier.
    """
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk", "risk-low"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"


def generate_recommendations(customer: dict, will_churn: bool, probability: float) -> list:
    """
    Generate business-oriented retention recommendations based on the
    prediction outcome and the customer's profile. Purely rule-based —
    not part of the model itself.
    """
    recs = []

    if not will_churn and probability < RISK_THRESHOLDS["low"]:
        recs.append("✅ Customer shows strong loyalty signals — maintain standard engagement.")
        recs.append("💡 Consider upselling premium add-ons (streaming, device protection).")
        return recs

    # Escalating retention actions for at-risk customers
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
        st.markdown("## 📉 Churn Prediction")
        st.markdown("### Business Dashboard")
        st.markdown("---")

        st.markdown("#### 📘 About This Project")
        st.markdown(
            "This dashboard predicts whether a telecom customer is likely "
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

def render_prediction_result(probability: float, will_churn: bool) -> None:
    """Render the result card and gauge chart for a single prediction."""
    risk_label, risk_class = get_risk_level(probability)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        card_class = "churn" if will_churn else "retain"
        title = "⚠️ Likely to Churn" if will_churn else "✅ Likely to Stay"
        display_prob = probability if will_churn else (1 - probability)
        prob_label = "Churn Probability" if will_churn else "Retention Probability"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">{title}</div>
                <div class="result-value">{display_prob:.1%}</div>
                <div>{prob_label}</div>
                <br/>
                <span class="risk-badge {risk_class}">{risk_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Predicted Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1a2b48"},
                    "steps": [
                        {"range": [0, 30], "color": "#bbf7d0"},
                        {"range": [30, 60], "color": "#fef08a"},
                        {"range": [60, 100], "color": "#fecaca"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": probability * 100,
                    },
                },
            )
        )
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render_recommendations(customer: dict, will_churn: bool, probability: float) -> None:
    """Render the business recommendations section."""
    st.markdown('<div class="section-title">💡 Business Recommendations</div>', unsafe_allow_html=True)
    recs = generate_recommendations(customer, will_churn, probability)
    for rec in recs:
        st.markdown(f'<div class="recommend-item">{rec}</div>', unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    configure_page()

    st.markdown('<div class="app-title">📉 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">A business intelligence tool for identifying at-risk '
        'customers and recommending retention actions — powered by a tuned Gradient Boosting model.</div>',
        unsafe_allow_html=True,
    )

    render_sidebar()

    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()  # Halt execution — cannot proceed without the model artifact

    with st.form("customer_input_form"):
        personal = render_personal_information()
        account = render_account_information()
        internet = render_internet_services()
        billing = render_billing_information(default_tenure=account["tenure"])

        submitted = st.form_submit_button("🔍 Predict Churn")

    if not submitted:
        st.info("👆 Fill in the customer details above and click **Predict Churn** to see the result.")
        return

    # Assemble full customer record
    customer = {**personal, **account, **internet, **billing}

    # ---- Input validation ----
    issues = validate_inputs(
        tenure=customer["tenure"],
        monthly_charges=customer["MonthlyCharges"],
        total_charges=customer["TotalCharges"],
    )
    if issues:
        for issue in issues:
            st.error(f"⚠️ {issue}")
        return

    # ---- Build DataFrame with exact column names/order expected by the pipeline ----
    try:
        input_df = pd.DataFrame([customer])[ALL_FEATURES]
    except KeyError as exc:
        st.error(f"❌ Column mismatch when building model input: {exc}")
        return

    # ---- Run prediction through the saved pipeline (no retraining/refitting) ----
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # P(Churn = Yes)
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Prediction failed: {exc}")
        return

    will_churn = bool(prediction == 1)

    st.markdown("---")
    st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
    render_prediction_result(probability, will_churn)

    st.markdown("---")
    render_recommendations(customer, will_churn, probability)

    with st.expander("🔎 View Model Input Data"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)


if __name__ == "__main__":
    main()
