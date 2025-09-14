import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
# -------------------------------
# Load Preprocessor and Model
# -------------------------------
pipeline = joblib.load("preprocessor_pipeline.pkl")
model = tf.keras.models.load_model("loan_approver_model.h5")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳", layout="centered")

st.title("💳 Loan Approval Prediction App")
st.write("Enter applicant details below to check loan eligibility.")

# -------------------------------
# Input Form
# -------------------------------
with st.form("loan_form"):
    st.subheader("Applicant Information")
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income ($)", 0, 150000, 10000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", 0, 250000, 5000, step=1000)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    months_employed = st.number_input("Months Employed", 0, 600, 24)
    num_credit_lines = st.number_input("Number of Credit Lines", 0, 50, 5)
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0, step=0.1)
    loan_term = st.number_input("Loan Term (Months)", 6, 360, 36)
    dti_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 2.0, 0.3, step=0.01)

    st.subheader("Additional Information")
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
    has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Other"])
    has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    # Collect input into DataFrame
    raw_input = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }
    raw_df = pd.DataFrame([raw_input])

    # Preprocess input
    processed_input = pipeline.transform(raw_df)

    # Predict probability
    prob = model.predict(processed_input)[0][0]
    prediction = 1 if prob >= 0.5 else 0

    # Display result
    st.subheader("📌 Prediction Result")
    if prediction == 0:
        st.success(f"✅ Loan Approved (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Loan Not Approved (Probability: {prob:.2f})")

