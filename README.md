# 📉 Customer Churn Prediction Dashboard

A production-ready Streamlit business dashboard that predicts customer churn
for a telecom company using a pre-trained, tuned **Gradient Boosting**
pipeline (scikit-learn).

> This app does **not** retrain or rebuild the model. It loads the already
> trained pipeline artifact (`customer_churn_pipeline.pkl`) — which includes
> both the fitted preprocessing steps (`StandardScaler`, `OneHotEncoder` via a
> `ColumnTransformer`) and the final `GradientBoostingClassifier` — and uses
> it strictly for inference.

---

## 🗂️ Project Structure

```
.
├── app.py                          # Streamlit application (entry point)
├── customer_churn_pipeline.pkl     # Pre-trained sklearn Pipeline (preprocessing + model)
├── requirements.txt                # Python dependencies (pinned for reproducibility)
└── README.md                       # This file
```

---

## 📘 About the Project

Customer churn — when a customer stops doing business with a company — is
one of the most costly problems for subscription-based businesses. This
dashboard lets a business user enter a customer's profile and instantly get:

- A **Churn / No Churn** prediction
- A **probability score** for how likely the customer is to churn
- A **risk tier** (Low / Medium / High)
- **Actionable retention recommendations** tailored to that customer's profile

## 🗃️ Dataset

- **Source:** Telco Customer Churn dataset (IBM sample dataset)
- **Size:** 7,043 customers, 19 input features
- **Target:** `Churn` (Yes / No)
- **Features cover:** demographics, account/contract details, subscribed
  services (internet, phone, streaming, security add-ons), and billing info.

## 🤖 Model

- **Algorithm:** Gradient Boosting Classifier
- **Hyperparameter tuning:** `GridSearchCV` over `n_estimators`,
  `learning_rate`, and `max_depth` (5-fold CV, scored on ROC AUC)
- **Preprocessing (baked into the pipeline):**
  - `StandardScaler` on numeric features: `SeniorCitizen`, `tenure`,
    `MonthlyCharges`, `TotalCharges`
  - `OneHotEncoder(handle_unknown="ignore")` on the remaining 15 categorical
    features
- **Reported performance (from the training notebook):**

  | Metric          | Value  |
  |-----------------|--------|
  | Test Accuracy   | 79.2%  |
  | Precision       | 64.7%  |
  | Recall          | 47.6%  |
  | F1 Score        | 54.9%  |
  | CV ROC AUC      | 0.851  |

The Gradient Boosting model was selected after comparing 9 candidate
algorithms (Logistic Regression, Decision Tree, Random Forest, Extra Trees,
AdaBoost, KNN, SVM, Naive Bayes, Gradient Boosting) via 5-fold cross
validation.

---

## ⚙️ Setup Instructions (Local)

1. **Clone / download this folder** and `cd` into it.

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Confirm the model file is present.** `customer_churn_pipeline.pkl`
   must sit in the same directory as `app.py`.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

6. Open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this folder (including `customer_churn_pipeline.pkl`) to a **public
   GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click
   **"New app"**.
3. Select your repository, branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will automatically install everything
   listed in `requirements.txt`.
5. No secrets, environment variables, or additional configuration are
   required — the app is self-contained.

> **Note on scikit-learn / numpy versions:** `requirements.txt` pins
> `scikit-learn==1.7.2` and `numpy>=2.0,<3.0` to match the versions the
> pipeline was originally trained and pickled with. A numpy `1.x` vs `2.x`
> mismatch between training and serving environments will cause errors like
> `"<class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator
> module"` when loading the pickle — this happens because internal random
> state objects inside the fitted model are serialized differently across
> numpy's major versions. If you retrain the model with different
> scikit-learn/numpy versions, update these pins to match.

---

## 🖥️ Using the App

1. Fill in the customer's details across the four sections:
   **Personal Information**, **Account Information**, **Internet Services**,
   and **Billing Information**.
2. Click **🔍 Predict Churn**.
3. Review:
   - The **prediction card** (Likely to Churn / Likely to Stay)
   - The **churn probability gauge**
   - The **risk badge** (Low / Medium / High Risk)
   - **Business recommendations** for retention actions
4. Expand **"View Model Input Data"** to inspect exactly what was sent to
   the model — useful for debugging or auditing.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [scikit-learn](https://scikit-learn.org/) — trained pipeline (preprocessing + model)
- [Plotly](https://plotly.com/python/) — interactive gauge chart
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

---

## 👨‍💻 Developer

**Name:** Utkarsh Singh Dahiya
**Role:** Data Scientist / ML Engineer

> Add your email and portfolio link here whenever you'd like them displayed.
