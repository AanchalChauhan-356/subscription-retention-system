import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Subscription Retention Intelligence System")

st.write("Enter customer details:")

# Create input fields dynamically
input_data = {}

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data[col] = val

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale input
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]

    if prob < 0.3:
        risk = "Low"
        advice = "Maintain customer satisfaction"
    elif prob < 0.7:
        risk = "Medium"
        advice = "Engage customer with offers"
    else:
        risk = "High"
        advice = "Offer discount or upgrade plan"

    st.success(f"Churn Probability: {prob:.2f}")
    st.success(f"Risk Level: {risk}")
    st.info(f"Recommendation: {advice}")S