import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Load Model, Scaler, and Columns ---
with open("C:/Users/pavan/Desktop/dataScience/Telco_Churn_Project/venv/model/LogisticRegressionModel", "rb") as f:
    model = pickle.load(f)

with open("C:/Users/pavan/Desktop/dataScience/Telco_Churn_Project/venv/model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("C:/Users/pavan/Desktop/dataScience/Telco_Churn_Project/venv/model/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")
st.title(" Telco Customer Churn Prediction App")
st.markdown("This app predicts whether a customer is likely to churn based on their service usage and demographics.")

# --- Input Fields ---
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# --- Create Input Dictionary ---
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

# --- Convert Input to DataFrame ---
input_df = pd.DataFrame([input_dict])

# --- One-Hot Encoding ---
input_encoded = pd.get_dummies(input_df)

# --- Handle Missing Columns ---
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# --- Ensure Column Order Matches Training Data ---
input_encoded = input_encoded[model_columns]

# --- Feature Scaling ---
# Copy original
input_scaled = input_encoded.copy()

# Scale only the numerical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_scaled[num_cols] = scaler.transform(input_scaled[num_cols])




# --- Prediction ---
if st.button(" Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # optional: confidence score

    if prediction == 1:
        st.error(f" The customer is likely to **churn**. (Confidence: {probability:.2%})")
    else:
        st.success(f" The customer is likely to **stay**. (Confidence: {probability:.2%})")



