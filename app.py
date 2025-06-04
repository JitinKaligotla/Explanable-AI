import streamlit as st
import pandas as pd
import pickle
import shap
import warnings
from streamlit_shap import st_shap

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

# === Load model components ===
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
    label_encoders.pop('Loan_Status', None)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("X_columns.pkl", "rb") as f:
    trained_columns = pickle.load(f)

with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# === Collect user input ===
def user_input_features():
    st.markdown("### 📝 Please enter your loan details:")

    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    married = st.selectbox("Married", label_encoders['Married'].classes_)
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3'])
    education = st.selectbox("Education", label_encoders['Education'].classes_)
    self_employed = st.selectbox("Self Employed", label_encoders['Self_Employed'].classes_)

    applicant_income = st.number_input("Applicant Income (₹)", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1)
    loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=1)
    credit_history = st.selectbox("Credit History", [0.0, 1.0])
    property_area = st.selectbox("Property Area", label_encoders['Property_Area'].classes_)

    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    return pd.DataFrame([data]), data

# === Preprocess input ===
def preprocess_input(df):
    for col in label_encoders:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col].astype(str))
    df['Dependents'] = df['Dependents'].astype(int)
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[num_cols] = scaler.transform(df[num_cols])
    return df[trained_columns]

# === App UI ===
st.title("🏦 Loan Approval Prediction")
st.markdown("Use this tool to check if your loan is likely to be approved.")

input_df, raw_input = user_input_features()
processed_df = preprocess_input(input_df)

if st.button("🔍 Predict"):
    prediction = model.predict(processed_df)[0]
    prediction_proba = model.predict_proba(processed_df)[0][prediction]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"

    st.success(f"### Loan Status: {result}  \nConfidence: **{prediction_proba:.2f}**")

    # === SHAP Explanation ===
    st.markdown("---")
    st.markdown("### 🔍 Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_df)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values

    shap_row = shap_vals[0]
    if hasattr(shap_row, 'ndim') and shap_row.ndim == 2 and shap_row.shape[1] == 2:
        shap_series = pd.Series(shap_row[:, 1], index=processed_df.columns)
    else:
        shap_series = pd.Series(shap_row, index=processed_df.columns)

    top_features = shap_series.abs().sort_values(ascending=False).head(3).index.tolist()

    st.markdown("#### 📊 Top contributing features:")
    for i, feat in enumerate(top_features, 1):
        st.markdown(f"- {i}. **{feat}**")

    st.markdown("#### 🗣️ Explanation in simple words:")

    explanation_sentences = []
    for feat in top_features:
        value = raw_input.get(feat, "N/A")
        shap_val = shap_series[feat]
        impact = "positively influenced" if shap_val > 0 else "negatively affected"

        if feat == 'ApplicantIncome':
            explanation_sentences.append(f"Your income of ₹{value} {impact} your chances of loan approval.")
        elif feat == 'CoapplicantIncome':
            explanation_sentences.append(f"Your coapplicant's income of ₹{value} {impact} your approval likelihood.")
        elif feat == 'LoanAmount':
            explanation_sentences.append(f"A requested loan amount of ₹{value} {impact} the risk profile.")
        elif feat == 'Credit_History':
            if shap_val > 0:
                explanation_sentences.append("Having a positive credit history improved your chances.")
            else:
                explanation_sentences.append("A missing or poor credit history reduced your chances.")
        elif feat == 'Loan_Amount_Term':
            explanation_sentences.append(f"The loan term of {value} months {impact} your approval chances.")
        elif feat == 'Education':
            explanation_sentences.append(f"Your education level {impact} your loan application outcome.")
        elif feat == 'Self_Employed':
            explanation_sentences.append(f"Being self-employed {impact} your approval likelihood.")
        elif feat == 'Property_Area':
            explanation_sentences.append(f"The location of your property {impact} the approval decision.")
        elif feat == 'Dependents':
            explanation_sentences.append(f"Having {value} dependents {impact} your chances.")
        else:
            explanation_sentences.append(f"{feat} {impact} your loan approval outcome.")

    for sentence in explanation_sentences:
        st.markdown(f"- {sentence}")
