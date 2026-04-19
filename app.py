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
st.markdown("Upload customer data and analyze churn with smart insights")

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

        # Convert to percentage
        df['Churn_Probability'] = (probs * 100).round(2)

        # Risk segmentation
        def risk(p):
            if p < 30:
                return "Low"
            elif p < 70:
                return "Medium"
            else:
                return "High"

        df['Risk_Level'] = df['Churn_Probability'].apply(risk)

        # Emoji labels
        def risk_label(r):
            if r == "Low":
                return "🟢 Low"
            elif r == "Medium":
                return "🟡 Medium"
            else:
                return "🔴 High"

        df['Risk_Display'] = df['Risk_Level'].apply(risk_label)

        # Recommendations
        def advice(r):
            if r == "Low":
                return "Maintain engagement"
            elif r == "Medium":
                return "Offer targeted promotions"
            else:
                return "Immediate retention action required"

        df['Recommendation'] = df['Risk_Level'].apply(advice)

        # -------- FILTERS --------
        st.sidebar.header("🔍 Filters")

        selected_risk = st.sidebar.multiselect(
            "Select Risk Level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

        prob_range = st.sidebar.slider(
            "Churn Probability (%)",
            0, 100, (0, 100)
        )

        if 'MonthlyCharges' in df.columns:
            charge_range = st.sidebar.slider(
                "Monthly Charges",
                float(df['MonthlyCharges'].min()),
                float(df['MonthlyCharges'].max()),
                (float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()))
            )
        else:
            charge_range = None

        # Apply filters
        filtered_df = df[
            (df['Risk_Level'].isin(selected_risk)) &
            (df['Churn_Probability'] >= prob_range[0]) &
            (df['Churn_Probability'] <= prob_range[1])
        ]

        if charge_range:
            filtered_df = filtered_df[
                (filtered_df['MonthlyCharges'] >= charge_range[0]) &
                (filtered_df['MonthlyCharges'] <= charge_range[1])
            ]

        # -------- KPI --------
        st.subheader("📊 Key Insights")

        col1, col2, col3 = st.columns(3)

        col1.metric("Customers", len(filtered_df))
        col2.metric("High Risk", (filtered_df['Risk_Level'] == "High").sum())
        col3.metric("Avg Churn %", f"{filtered_df['Churn_Probability'].mean():.2f}%")

        # -------- TABLE --------
        st.subheader("📋 Filtered Results")

        display_df = filtered_df.drop(columns=['Risk_Level'])
        st.dataframe(display_df)

        # -------- HIGH RISK --------
        st.subheader("🚨 High Risk Customers")

        high_df = filtered_df[filtered_df['Risk_Level'] == "High"]

        if len(high_df) > 0:
            st.error(f"{len(high_df)} customers need immediate attention!")
            st.dataframe(high_df.head(10))
        else:
            st.success("No high-risk customers 🎉")

        # -------- DASHBOARD --------
        st.subheader("📊 Visual Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.pie(filtered_df, names='Risk_Display', hole=0.5,
                          title="Risk Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(filtered_df, x='Churn_Probability',
                                title="Churn Probability (%)")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            risk_counts = filtered_df['Risk_Display'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Level', 'Count']

            fig3 = px.bar(risk_counts, x='Risk_Level', y='Count',
                          color='Risk_Level', title="Risk Count")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            if 'MonthlyCharges' in filtered_df.columns:
                fig4 = px.scatter(filtered_df,
                                  x='MonthlyCharges',
                                  y='Churn_Probability',
                                  color='Risk_Display',
                                  title="Charges vs Churn %")
                st.plotly_chart(fig4, use_container_width=True)

        # -------- DOWNLOAD --------
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
