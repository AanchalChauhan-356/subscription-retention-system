import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Load model files
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
columns_path = os.path.join(os.path.dirname(__file__), "columns.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
columns = pickle.load(open(columns_path, "rb"))

st.title("📊 Subscription Retention Intelligence System")

st.write("Upload customer data (CSV) to predict churn risk")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data")
    st.dataframe(df.head())

    try:
    import plotly.express as px

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    if 'Churn' in df.columns:
        df.drop('Churn', axis=1, inplace=True)

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    scaled_data = scaler.transform(df)

    # Prediction
    probs = model.predict_proba(scaled_data)[:, 1]
    df['Churn_Probability'] = probs

    def risk(p):
        if p < 0.3:
            return "Low"
        elif p < 0.7:
            return "Medium"
        else:
            return "High"

    df['Risk_Level'] = df['Churn_Probability'].apply(risk)

    def advice(r):
        if r == "Low":
            return "Maintain satisfaction"
        elif r == "Medium":
            return "Engage with offers"
        else:
            return "Give discounts / retention plan"

    df['Recommendation'] = df['Risk_Level'].apply(advice)

    st.write("### ✅ Predictions")
    st.dataframe(df)

    # ================= DASHBOARD =================

    st.write("## 📊 Dashboard")

    # Donut chart
    st.write("### 🎯 Risk Distribution")
    fig1 = px.pie(df, names='Risk_Level', hole=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    # Histogram
    st.write("### 📈 Churn Probability Distribution")
    fig2 = px.histogram(df, x='Churn_Probability', nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

    # Bar chart
    st.write("### 📊 Risk Count")
    risk_counts = df['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Level', 'Count']

    fig3 = px.bar(risk_counts, x='Risk_Level', y='Count', color='Risk_Level')
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter
    if 'MonthlyCharges' in df.columns:
        st.write("### 💡 Charges vs Churn Risk")
        fig4 = px.scatter(df, x='MonthlyCharges', y='Churn_Probability', color='Risk_Level')
        st.plotly_chart(fig4, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
