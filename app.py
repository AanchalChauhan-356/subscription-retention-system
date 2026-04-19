import streamlit as st
import pickle
import pandas as pd
import os
import plotly.express as px

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
columns_path = os.path.join(os.path.dirname(__file__), "columns.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
columns = pickle.load(open(columns_path, "rb"))

# Page config
st.set_page_config(page_title="Churn Intelligence", layout="wide")

# Title
st.title("🚀 Subscription Retention Intelligence System")
st.markdown("Upload customer data and get churn insights + recommendations")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    try:
        # -------- PREPROCESS --------
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)

        if 'Churn' in df.columns:
            df.drop('Churn', axis=1, inplace=True)

        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        scaled = scaler.transform(df)

        # -------- PREDICT --------
        probs = model.predict_proba(scaled)[:, 1]
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
                return "✅ Maintain engagement (loyal customers)"
            elif r == "Medium":
                return "⚠️ Offer targeted promotions"
            else:
                return "🚨 Immediate retention action (discounts / support)"

        df['Recommendation'] = df['Risk_Level'].apply(advice)

        # -------- KPI CARDS --------
        st.subheader("📊 Key Insights")

        col1, col2, col3 = st.columns(3)

        total = len(df)
        high_risk = (df['Risk_Level'] == "High").sum()
        avg_prob = df['Churn_Probability'].mean()

        col1.metric("Total Customers", total)
        col2.metric("High Risk Customers", high_risk)
        col3.metric("Avg Churn Probability", f"{avg_prob:.2f}")

        # -------- TABLE --------
        st.subheader("📋 Prediction Results")
        st.dataframe(df)

        # -------- HIGH RISK ALERT --------
        st.subheader("🚨 High Risk Customers")

        high_df = df[df['Risk_Level'] == "High"]

        if len(high_df) > 0:
            st.error(f"{len(high_df)} customers are at HIGH risk!")
            st.dataframe(high_df.head(10))
        else:
            st.success("No high-risk customers 🎉")

        # -------- DASHBOARD --------
        st.subheader("📊 Visual Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.pie(df, names='Risk_Level', hole=0.5,
                          title="Risk Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(df, x='Churn_Probability', nbins=30,
                                title="Probability Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Level', 'Count']

            fig3 = px.bar(risk_counts, x='Risk_Level', y='Count',
                          color='Risk_Level', title="Risk Count")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            if 'MonthlyCharges' in df.columns:
                fig4 = px.scatter(df,
                                  x='MonthlyCharges',
                                  y='Churn_Probability',
                                  color='Risk_Level',
                                  title="Charges vs Risk")
                st.plotly_chart(fig4, use_container_width=True)

        # -------- DOWNLOAD --------
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
