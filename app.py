"""
Customer Churn Prediction — Business Intelligence Dashboard
=============================================================
A production-quality Streamlit application that uses a pre-trained
scikit-learn pipeline (ColumnTransformer + GradientBoostingClassifier)
to predict customer churn for a telecom business.

Author  : (Your Name)
Model   : GradientBoostingClassifier (tuned via GridSearchCV)
Dataset : Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)

IMPORTANT
---------
This app does NOT retrain or reconstruct the preprocessing pipeline.
It simply loads `customer_churn_pipeline.pkl` (already contains the
ColumnTransformer + StandardScaler + OneHotEncoder + final model) and
uses it directly for inference.

NOTE ON THEMING
---------------
The app ships with `.streamlit/config.toml`, which locks in a single
dark, professional theme. This guarantees the dashboard looks identical
for every visitor regardless of their browser/OS light-dark preference
— avoiding the mismatched "white card on dark background" problem that
happens when CSS assumes one theme but the visitor's client renders
another.
"""

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Customer Churn Intelligence | Predictive Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS — must exactly mirror the categories seen during training
# (extracted directly from the fitted OneHotEncoder inside the pipeline)
# =============================================================================
MODEL_PATH = Path(__file__).parent / "customer_churn_pipeline.pkl"

GENDER_OPTIONS = ["Female", "Male"]
YES_NO_OPTIONS = ["No", "Yes"]
MULTIPLE_LINES_OPTIONS = ["No", "No phone service", "Yes"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
INTERNET_DEPENDENT_OPTIONS = ["No", "No internet service", "Yes"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

# The exact 19 feature columns the pipeline was fitted on (order-agnostic,
# ColumnTransformer selects by name, but we keep this for validation/reference)
EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# =============================================================================
# CUSTOM CSS — professional, dark, business-dashboard styling
# =============================================================================
def load_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hide default Streamlit chrome for a cleaner dashboard feel */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}

        .stApp {
            background: radial-gradient(circle at top left, #10202f 0%, #0b1420 55%, #090f18 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* ---------- Page header ---------- */
        .app-header {
            background: linear-gradient(135deg, #103246 0%, #17435a 50%, #1d5b73 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.8rem;
            border: 1px solid rgba(79, 209, 197, 0.25);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
        }
        .app-header h1 {
            color: #ffffff;
            font-weight: 800;
            font-size: 2.05rem;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .app-header p {
            color: #cfe8ef;
            font-size: 1.02rem;
            margin-top: 0.5rem;
            margin-bottom: 0;
            font-weight: 400;
        }
        .app-header .badge {
            display: inline-block;
            background: rgba(79, 209, 197, 0.14);
            color: #b7f5ec;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.9rem;
            margin-right: 0.5rem;
            border: 1px solid rgba(79, 209, 197, 0.35);
        }

        /* ---------- Section containers ----------
           These target the native st.container(key=...) wrapper so the
           border/background fully encloses the title AND every widget
           inside it (fields no longer float outside the card). */
        div[class*="st-key-section_"] {
            background: #101c2b;
            border: 1px solid #223547 !important;
            border-radius: 14px !important;
            padding: 0.4rem 1.3rem 1.1rem 1.3rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 2px 14px rgba(0, 0, 0, 0.25);
        }

        .section-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #eef4f8;
            margin: 0.9rem 0 1rem 0;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid #223547;
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }

        /* ---------- Form field labels ---------- */
        label, .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #b9c6d4 !important;
            font-weight: 500 !important;
        }

        /* Selectbox / number input surfaces */
        div[data-baseweb="select"] > div,
        .stNumberInput input {
            background-color: #16283b !important;
            border: 1px solid #2b3f52 !important;
            color: #eef4f8 !important;
            border-radius: 8px !important;
        }

        /* ---------- Predict button ---------- */
        div.stButton > button,
        div.stFormSubmitButton > button {
            background: linear-gradient(135deg, #17435a 0%, #1d5b73 60%, #23788f 100%);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            border: 1px solid rgba(79, 209, 197, 0.4);
            width: 100%;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(35, 120, 143, 0.45);
            border: 1px solid rgba(79, 209, 197, 0.7);
            color: white;
        }

        /* ---------- Result cards ---------- */
        .result-card {
            border-radius: 16px;
            padding: 1.8rem 2rem;
            margin-top: 1rem;
            margin-bottom: 1.2rem;
        }
        .result-card-danger {
            background: linear-gradient(135deg, #2b1414 0%, #3a1818 100%);
            border: 1px solid #6b2b2b;
        }
        .result-card-success {
            background: linear-gradient(135deg, #10261b 0%, #123322 100%);
            border: 1px solid #2f6b47;
        }
        .result-title-danger { color: #ff8080; font-size: 1.5rem; font-weight: 800; margin: 0;}
        .result-title-success { color: #6fe0a0; font-size: 1.5rem; font-weight: 800; margin: 0;}
        .result-sub { color: #c3d0da; font-size: 0.95rem; margin-top: 0.4rem; }

        /* ---------- Risk badges ---------- */
        .risk-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.3px;
            margin-top: 0.9rem;
        }
        .risk-high { background: rgba(255, 99, 99, 0.15); color: #ff8080; border: 1px solid rgba(255, 99, 99, 0.4); }
        .risk-medium { background: rgba(245, 176, 65, 0.15); color: #f5b041; border: 1px solid rgba(245, 176, 65, 0.4); }
        .risk-low { background: rgba(88, 214, 141, 0.15); color: #6fe0a0; border: 1px solid rgba(88, 214, 141, 0.4); }

        /* ---------- Metric tiles ---------- */
        .metric-tile {
            background: #101c2b;
            border: 1px solid #223547;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-tile .label {
            font-size: 0.78rem;
            color: #8fa2b3;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-tile .value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #eef4f8;
            margin-top: 0.2rem;
        }

        /* ---------- Recommendation box ---------- */
        .rec-box {
            background: #101c2b;
            border-left: 4px solid #23788f;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.6rem;
            font-size: 0.94rem;
            color: #dbe6ef;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #0d1824;
            border-right: 1px solid #1b2a38;
        }
        section[data-testid="stSidebar"] * {
            color: #d3dee8 !important;
        }
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] a {
            color: #4fd1c5 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-top: 1px solid #1b2a38;
        }

        /* Expander */
        details {
            background: #101c2b !important;
            border: 1px solid #223547 !important;
            border-radius: 10px !important;
        }
        summary {
            color: #eef4f8 !important;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            border: 1px solid #223547;
            border-radius: 10px;
        }

        /* Generic markdown text on main page */
        .stMarkdown, .stMarkdown p {
            color: #dbe6ef;
        }

        hr {
            border: none;
            border-top: 1px solid #223547;
            margin: 1.2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MODEL LOADING (cached — loaded once per session)
# =============================================================================
@st.cache_resource(show_spinner="Loading trained pipeline...")
def load_pipeline(model_path: Path):
    """Load the pre-trained, pre-fitted sklearn pipeline (.pkl).

    The pipeline already bundles preprocessing (StandardScaler +
    OneHotEncoder inside a ColumnTransformer) and the tuned
    GradientBoostingClassifier. We never rebuild or retrain it here.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Make sure 'customer_churn_pipeline.pkl' is in the app directory."
        )
    return joblib.load(model_path)


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📡 About This Project")
        st.markdown(
            "An end-to-end **customer churn prediction system** built for "
            "telecom subscription businesses, allowing teams to identify "
            "at-risk customers before they cancel their service."
        )

        st.markdown("---")
        st.markdown("### 🗂️ Dataset")
        st.markdown(
            "**Telco Customer Churn** dataset — 7,043 customer records "
            "covering demographics, account tenure, subscribed services, "
            "contract type, and billing details."
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Details")
        st.markdown("**Algorithm:** Gradient Boosting Classifier")
        st.markdown("**Tuning:** GridSearchCV (5-fold CV, ROC-AUC scoring)")
        st.markdown("**Preprocessing:** `ColumnTransformer`")
        st.markdown("&nbsp;&nbsp;• `StandardScaler` → numeric features", unsafe_allow_html=True)
        st.markdown("&nbsp;&nbsp;• `OneHotEncoder` → categorical features", unsafe_allow_html=True)
        st.markdown(
            "**Artifact:** single serialized `Pipeline` object "
            "(preprocessing + model), loaded via `joblib`"
        )

        st.markdown("---")
        st.markdown("### ⚙️ How It Works")
        st.markdown("1. Enter customer details in the form")
        st.markdown("2. Inputs are assembled into a single-row DataFrame")
        st.markdown("3. The saved pipeline transforms & scores the data")
        st.markdown("4. Churn probability and risk level are returned instantly")

        st.markdown("---")
        st.markdown("### 👤 Developer")
        st.markdown("**Your Name**")
        st.markdown("AI/ML Engineer · Data Scientist")
        st.markdown("[LinkedIn](#) · [GitHub](#) · [Portfolio](#)")

        st.markdown("---")
        st.caption("Built with Streamlit · scikit-learn · joblib")


# =============================================================================
# INPUT FORM SECTIONS
# Each section uses st.container(border=True, key=...) so the CSS above can
# style ONE box that fully encloses the title and all its widgets together.
# =============================================================================
def render_personal_information() -> dict:
    with st.container(key="section_personal"):
        st.markdown(
            '<div class="section-title">👤 Personal Information</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", GENDER_OPTIONS, help="Customer's gender")
        with c2:
            senior_citizen_label = st.selectbox(
                "Senior Citizen", ["No", "Yes"], help="Is the customer 65 years or older?"
            )
        with c3:
            partner = st.selectbox("Has Partner", YES_NO_OPTIONS)

        c4, _, _ = st.columns(3)
        with c4:
            dependents = st.selectbox("Has Dependents", YES_NO_OPTIONS)

    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
    }


def render_account_information() -> dict:
    with st.container(key="section_account"):
        st.markdown(
            '<div class="section-title">📑 Account Information</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider(
                "Tenure (months)", min_value=0, max_value=72, value=12, step=1,
                help="Number of months the customer has stayed with the company",
            )
        with c2:
            contract = st.selectbox("Contract Type", CONTRACT_OPTIONS)
        with c3:
            paperless_billing = st.selectbox("Paperless Billing", YES_NO_OPTIONS)

        c4, _, _ = st.columns(3)
        with c4:
            payment_method = st.selectbox("Payment Method", PAYMENT_METHOD_OPTIONS)

    return {
        "tenure": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }


def render_internet_services() -> dict:
    with st.container(key="section_internet"):
        st.markdown(
            '<div class="section-title">🌐 Internet & Add-on Services</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            phone_service = st.selectbox("Phone Service", YES_NO_OPTIONS)
        with c2:
            # MultipleLines only truly applies if PhoneService = Yes, but we keep
            # the widget available and let the model's OneHotEncoder(handle_unknown)
            # handle any combination the user selects.
            multiple_lines = st.selectbox("Multiple Lines", MULTIPLE_LINES_OPTIONS)
        with c3:
            internet_service = st.selectbox("Internet Service", INTERNET_SERVICE_OPTIONS)

        c4, c5, c6 = st.columns(3)
        with c4:
            online_security = st.selectbox("Online Security", INTERNET_DEPENDENT_OPTIONS)
        with c5:
            online_backup = st.selectbox("Online Backup", INTERNET_DEPENDENT_OPTIONS)
        with c6:
            device_protection = st.selectbox("Device Protection", INTERNET_DEPENDENT_OPTIONS)

        c7, c8, c9 = st.columns(3)
        with c7:
            tech_support = st.selectbox("Tech Support", INTERNET_DEPENDENT_OPTIONS)
        with c8:
            streaming_tv = st.selectbox("Streaming TV", INTERNET_DEPENDENT_OPTIONS)
        with c9:
            streaming_movies = st.selectbox("Streaming Movies", INTERNET_DEPENDENT_OPTIONS)

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


def render_billing_information() -> dict:
    with st.container(key="section_billing"):
        st.markdown(
            '<div class="section-title">💳 Billing Information</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=500.0,
                value=65.0, step=0.5, format="%.2f",
                help="The amount charged to the customer monthly",
            )
        with c2:
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=15000.0,
                value=780.0, step=1.0, format="%.2f",
                help="The total amount charged to the customer over their tenure",
            )

    return {
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


# =============================================================================
# VALIDATION
# =============================================================================
def validate_inputs(data: dict) -> list:
    """Run business-logic sanity checks on the collected inputs.
    Returns a list of human-readable warning/error messages."""
    issues = []

    if data["TotalCharges"] < data["MonthlyCharges"] and data["tenure"] > 1:
        issues.append(
            "Total Charges is lower than Monthly Charges despite tenure > 1 month. "
            "Please double-check these values."
        )

    if data["tenure"] == 0 and data["TotalCharges"] > 0:
        issues.append(
            "Tenure is 0 months but Total Charges is greater than 0. "
            "Please verify these figures."
        )

    if data["PhoneService"] == "No" and data["MultipleLines"] not in ("No phone service",):
        issues.append(
            "Phone Service is 'No' but Multiple Lines is not set to "
            "'No phone service'. Please correct this for consistency."
        )

    if data["InternetService"] == "No":
        internet_dependent_fields = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        wrong_fields = [
            f for f in internet_dependent_fields
            if data[f] != "No internet service"
        ]
        if wrong_fields:
            issues.append(
                "Internet Service is 'No', so dependent services "
                f"({', '.join(wrong_fields)}) should be set to "
                "'No internet service' for consistency."
            )

    return issues


# =============================================================================
# PREDICTION HELPERS
# =============================================================================
def get_risk_level(probability: float) -> tuple:
    """Map churn probability to a (label, css_class) risk tier."""
    if probability >= 0.66:
        return "High Risk", "risk-high"
    elif probability >= 0.33:
        return "Medium Risk", "risk-medium"
    else:
        return "Low Risk", "risk-low"


def get_recommendations(will_churn: bool, probability: float, data: dict) -> list:
    """Generate simple, rule-based business recommendations."""
    recs = []

    if will_churn:
        recs.append("📞 Prioritize proactive outreach — assign to the retention team within 48 hours.")
        if data["Contract"] == "Month-to-month":
            recs.append("📝 Offer an incentive to upgrade to a 1- or 2-year contract to improve lock-in.")
        if data["InternetService"] == "Fiber optic" and data["TechSupport"] == "No":
            recs.append("🛠️ Bundle a free Tech Support add-on for the first 3 months to boost satisfaction.")
        if data["PaymentMethod"] == "Electronic check":
            recs.append("💳 Encourage a switch to automatic payments — often correlated with lower churn.")
        if probability >= 0.8:
            recs.append("🚨 Very high churn probability — consider a personal call from account management.")
        if data["MonthlyCharges"] > 80:
            recs.append("💰 Review pricing plan; a loyalty discount may reduce cost-driven churn.")
    else:
        recs.append("✅ Customer appears stable — continue standard engagement and satisfaction surveys.")
        if data["tenure"] < 6:
            recs.append("🌱 Still early in the lifecycle — nurture with onboarding and welcome offers.")
        recs.append("📈 Consider upsell opportunities (streaming add-ons, device protection) to grow revenue.")

    return recs


# =============================================================================
# MAIN APP
# =============================================================================
def main() -> None:
    load_custom_css()

    # ---------------- Header ----------------
    st.markdown(
        """
        <div class="app-header">
            <h1>📡 Customer Churn Intelligence Dashboard</h1>
            <p>Predict subscriber churn risk in real time using a tuned Gradient Boosting model.</p>
            <span class="badge">🤖 Gradient Boosting</span>
            <span class="badge">⚡ Real-time Scoring</span>
            <span class="badge">📊 Business-ready Insights</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    # ---------------- Load pipeline ----------------
    try:
        pipeline = load_pipeline(MODEL_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load the trained pipeline: {e}")
        st.stop()

    # ---------------- Input form ----------------
    with st.form("churn_prediction_form"):
        personal = render_personal_information()
        account = render_account_information()
        internet = render_internet_services()
        billing = render_billing_information()

        submitted = st.form_submit_button("🔍 Predict Churn Risk")

    # ---------------- Prediction ----------------
    if submitted:
        customer_data = {**personal, **account, **internet, **billing}

        # Validate business-logic consistency
        warnings_list = validate_inputs(customer_data)
        if warnings_list:
            for w in warnings_list:
                st.warning(f"⚠️ {w}")

        try:
            # Build a single-row DataFrame with EXACT column names expected
            # by the saved pipeline's ColumnTransformer.
            input_df = pd.DataFrame([customer_data])[EXPECTED_COLUMNS]

            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]  # P(Churn = Yes)

        except KeyError as e:
            st.error(
                f"❌ Missing expected feature column: {e}. "
                "Please make sure all form fields were completed."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred while generating the prediction: {e}")
            st.stop()

        will_churn = bool(prediction == 1)
        risk_label, risk_class = get_risk_level(probability)
        prob_pct = probability * 100

        st.markdown("---")

        # ---------------- Result card ----------------
        if will_churn:
            st.markdown(
                f"""
                <div class="result-card result-card-danger">
                    <p class="result-title-danger">⚠️ High Likelihood of Churn</p>
                    <p class="result-sub">This customer is predicted to <strong>cancel</strong> their subscription.</p>
                    <span class="risk-badge {risk_class}">{risk_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-card-success">
                    <p class="result-title-success">✅ Customer Likely to Stay</p>
                    <p class="result-sub">This customer is predicted to <strong>remain</strong> subscribed.</p>
                    <span class="risk-badge {risk_class}">{risk_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------- Metric tiles ----------------
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""<div class="metric-tile">
                        <div class="label">Churn Prediction</div>
                        <div class="value">{"Yes" if will_churn else "No"}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-tile">
                        <div class="label">Churn Probability</div>
                        <div class="value">{prob_pct:.1f}%</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class="metric-tile">
                        <div class="label">Risk Level</div>
                        <div class="value">{risk_label.replace(" Risk", "")}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        st.progress(min(max(probability, 0.0), 1.0))

        # ---------------- Recommendations ----------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💡 Recommended Business Actions")
        recommendations = get_recommendations(will_churn, probability, customer_data)
        for rec in recommendations:
            st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)

        # ---------------- Raw input (for transparency) ----------------
        with st.expander("🔎 View Input Data Sent to the Model"):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
