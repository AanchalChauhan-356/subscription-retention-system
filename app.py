import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

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

        # Risk Levels
        def risk(p):
            if p < 0.3:
                return "Low"
            elif p < 0.7:
                return "Medium"
            else:
                return "High"

        df['Risk_Level'] = df['Churn_Probability'].apply(risk)

        # Recommendations
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

        # 📊 DASHBOARD VISUALS

        st.write("## 📊 Dashboard")

        # 1️⃣ Risk Distribution
        st.write("### Risk Category Distribution")
        risk_counts = df['Risk_Level'].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%')
        st.pyplot(fig1)

        # 2️⃣ Probability Histogram
        st.write("### Churn Probability Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['Churn_Probability'], bins=20)
        ax2.set_xlabel("Churn Probability")
        ax2.set_ylabel("Number of Customers")
        st.pyplot(fig2)

        # 3️⃣ Risk Count Bar Chart
        st.write("### Risk Level Count")
        fig3, ax3 = plt.subplots()
        risk_counts.plot(kind='bar', ax=ax3)
        ax3.set_xlabel("Risk Level")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
