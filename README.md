# 📡 Customer Churn Intelligence Dashboard

A production-quality, business-facing Streamlit application that predicts telecom
customer churn using a pre-trained scikit-learn pipeline (`ColumnTransformer` +
`StandardScaler` + `OneHotEncoder` + tuned `GradientBoostingClassifier`).

This project is built as a portfolio-ready deployment layer on top of an existing,
already-trained model — **no retraining or preprocessing logic is duplicated here**.
The app simply loads `customer_churn_pipeline.pkl` and uses it for inference.

---

## ✨ Features

- Clean, modern business-dashboard UI (custom CSS, no default Streamlit look)
- Inputs organized into logical sections: **Personal, Account, Internet Services,
  Billing**
- Appropriate widgets for each feature type (selectbox for categorical, slider /
  number input for numerical)
- Client-side business-logic validation (e.g. tenure vs. total charges consistency)
- Real-time churn prediction with probability score
- Risk tiering (**Low / Medium / High**) with color-coded badges
- Rule-based business recommendations tailored to the prediction
- Sidebar with project, dataset, and model documentation
- Fully modular, commented, and easy to extend

---

## 🗂️ Project Structure

```
.
├── app.py                          # Main Streamlit application
├── customer_churn_pipeline.pkl     # Pre-trained pipeline (preprocessing + model)
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Locked-in dark theme (same look for every visitor)
└── README.md                       # This file
```

> **Keep the `.streamlit` folder!** It locks in a single professional dark
> theme so the dashboard looks identical for every visitor, regardless of
> their browser or OS light/dark setting. Without it, Streamlit falls back
> to the visitor's system theme, which can clash with the app's custom CSS.

---

## 🤖 Model Details

| Component        | Detail                                                        |
|-------------------|----------------------------------------------------------------|
| Preprocessing     | `ColumnTransformer` → `StandardScaler` (numeric) + `OneHotEncoder` (categorical) |
| Model             | `GradientBoostingClassifier` (tuned via `GridSearchCV`, 5-fold CV, ROC-AUC scoring) |
| Numeric features  | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical features | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Artifact          | Single serialized `sklearn.pipeline.Pipeline` object saved via `joblib.dump()` |

The app builds a single-row `pandas.DataFrame` from the user's form inputs using
**exactly** the column names the pipeline was fitted on, then calls
`pipeline.predict()` / `pipeline.predict_proba()` directly — the saved pipeline
handles all scaling and encoding internally.

---

## 🚀 Local Setup

1. **Clone or download this folder**, keeping `app.py`, `customer_churn_pipeline.pkl`,
   and `requirements.txt` together in the same directory.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in your terminal (typically `http://localhost:8501`).

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this folder (with `app.py`, `customer_churn_pipeline.pkl`, and
   `requirements.txt`) to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repository, branch, and set the main file
   path to `app.py`.
4. Click **"Deploy"** — Streamlit Cloud will automatically install the
   dependencies from `requirements.txt` and launch the app.

No code changes are required — the app is deployment-ready as-is.

> **Note on scikit-learn versions:** the pickle file was originally created with
> scikit-learn 1.7.2. `requirements.txt` pins a closely compatible version
> (1.8.0) that has been verified to load and score correctly. If you retrain
> the model with a different scikit-learn version, update this pin to match.

---

## 🧪 How Predictions Work

1. User fills out the form across four sections (Personal, Account, Internet
   Services, Billing).
2. On submit, inputs are validated for basic business-logic consistency
   (e.g. Phone Service = "No" should imply Multiple Lines = "No phone service").
3. Inputs are assembled into a single-row `DataFrame` with the exact column
   names expected by the saved pipeline.
4. `pipeline.predict()` returns the churn label (0 = No, 1 = Yes) and
   `pipeline.predict_proba()` returns the churn probability.
5. The probability is mapped to a risk tier:
   - **Low Risk:** probability < 33%
   - **Medium Risk:** 33% ≤ probability < 66%
   - **High Risk:** probability ≥ 66%
6. Business recommendations are generated based on the prediction and key
   customer attributes (contract type, tenure, payment method, etc.).

---

## 🛠️ Tech Stack

- **Streamlit** — UI framework
- **scikit-learn** — trained pipeline (preprocessing + model)
- **pandas / numpy** — data handling
- **joblib** — model serialization / deserialization

---

## 👤 Developer

**Utkarsh Singh Dahiya**

[AI/ML Engineer · Data Scientist]

---

## 📄 License

This project is provided for portfolio and educational purposes.
