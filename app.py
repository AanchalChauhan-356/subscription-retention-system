import streamlit as st
import pickle
import pandas as pd
import os

# Load model files
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
columns_path = os.path.join(os.path.dirname(__file__), "columns.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
columns = pickle.load(open(columns_path, "rb"))

st.title("📊 Subscription Retention Intelligence System")

st.write("Upload customer data (CSV) to predict churn risk")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data")
    st.dataframe(df.head())

    try:
        # PREPROCESS SAME AS TRAINING
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)

        # Convert target if exists
        if 'Churn' in df.columns:
            df.drop('Churn', axis=1, inplace=True)

        # One-hot encoding
        df = pd.get_dummies(df)

        # Align columns with training
        df = df.reindex(columns=columns, fill_value=0)

        # Scale
        scaled_data = scaler.transform(df)

        # Predict
        probs = model.predict_proba(scaled_data)[:, 1]

        # Add results
        df['Churn_Probability'] = probs

        # Risk categories
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

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
