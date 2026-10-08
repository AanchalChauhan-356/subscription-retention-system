import os
import pickle
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
st.set_page_config(page_title="Churn Intelligence", layout="wide")
 
# ---------------- LOAD MODEL FILES ----------------
BASE = os.path.dirname(__file__)
 
 
@st.cache_resource
def load_files():
    model = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))
    columns = pickle.load(open(os.path.join(BASE, "columns.pkl"), "rb"))
    return model, scaler, columns
 
 
model, scaler, columns = load_files()
 
# ---------------- HELPERS ----------------
# The model was trained on one-hot encoded columns like "Contract_One year".
# The CSV the user uploads has the ORIGINAL columns, like "Contract".
# So we work out which original columns the file must contain.
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
 
 
def required_raw_columns(model_columns):
    needed = []
    for c in model_columns:
        raw = c.split("_")[0]
        if raw not in needed:
            needed.append(raw)
    return needed
 
 
REQUIRED = required_raw_columns(columns)
 
SAMPLE_CSV = pd.DataFrame({
    "customerID": ["7590-VHVEG", "5575-GNVDE", "9237-HQITU"],
    "gender": ["Female", "Male", "Female"],
    "SeniorCitizen": [0, 0, 0],
    "Partner": ["Yes", "No", "No"],
    "Dependents": ["No", "No", "No"],
    "tenure": [1, 34, 2],
    "PhoneService": ["No", "Yes", "Yes"],
    "MultipleLines": ["No phone service", "No", "No"],
    "InternetService": ["DSL", "DSL", "Fiber optic"],
    "OnlineSecurity": ["No", "Yes", "No"],
    "OnlineBackup": ["Yes", "No", "No"],
    "DeviceProtection": ["No", "Yes", "No"],
    "TechSupport": ["No", "No", "No"],
    "StreamingTV": ["No", "No", "No"],
    "StreamingMovies": ["No", "No", "No"],
    "Contract": ["Month-to-month", "One year", "Month-to-month"],
    "PaperlessBilling": ["Yes", "No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
    "MonthlyCharges": [29.85, 56.95, 70.70],
    "TotalCharges": [29.85, 1889.50, 151.65],
}).to_csv(index=False).encode("utf-8")
 
 
def clean_upload(df):
    """Check the file has the right columns and tidy the values.
    Returns (clean_df, list_of_warnings). Raises ValueError with a friendly message."""
    df = df.copy()
    df.columns = df.columns.str.strip()
 
    if df.empty:
        raise ValueError("The file has no rows.")
 
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Your file is missing these columns: **" + ", ".join(missing) + "**. "
            "This model was trained on telecom customer data, so the file needs the "
            "same columns. Download the sample file below to see the format."
        )
 
    warnings = []
 
    # SeniorCitizen sometimes arrives as Yes/No instead of 1/0
    if not pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0})
 
    # Numeric columns: convert text/blank to numbers, fill gaps with the median
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = int(df[col].isna().sum())
        if bad:
            if df[col].notna().sum() == 0:
                raise ValueError(f"Column **{col}** has no valid numbers.")
            df[col] = df[col].fillna(df[col].median())
            warnings.append(f"{bad} blank or invalid value(s) in '{col}' were filled with the median.")
 
    return df, warnings
 
 
def predict(df):
    """Turn the cleaned upload into the exact shape the model expects."""
    X = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])
    X = pd.get_dummies(X)
    X = X.reindex(columns=columns, fill_value=0)
    return model.predict_proba(scaler.transform(X))[:, 1]
 
 
def risk(p):
    if p < 30:
        return "Low"
    elif p < 70:
        return "Medium"
    return "High"
 
 
ADVICE = {
    "Low": "Maintain engagement",
    "Medium": "Offer targeted promotions",
    "High": "Immediate retention action required",
}
EMOJI = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
 
# ---------------- PAGE ----------------
st.title("🚀 Subscription Retention Intelligence System")
st.markdown("Upload customer data and analyze churn with smart insights")
 
with st.expander("📋 What should my file look like?", expanded=False):
    st.write("The CSV needs these columns (extra columns like customerID are fine):")
    st.code(", ".join(REQUIRED))
    st.download_button("⬇️ Download sample CSV", SAMPLE_CSV, "sample_customers.csv", "text/csv")
 
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
 
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Couldn't read that file. Please upload a valid CSV.")
        st.stop()
 
    try:
        df, warnings = clean_upload(raw)
    except ValueError as e:
        st.error(str(e))
        st.download_button("⬇️ Download sample CSV", SAMPLE_CSV, "sample_customers.csv", "text/csv",
                           key="sample_after_error")
        st.stop()
 
    for w in warnings:
        st.warning(w)
 
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())
 
    try:
        probs = predict(df)
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")
        st.stop()
 
    # Results keep the ORIGINAL columns, not the one-hot encoded ones
    df["Churn_Probability"] = (probs * 100).round(2)
    df["Risk_Level"] = df["Churn_Probability"].apply(risk)
    df["Risk_Display"] = df["Risk_Level"].map(EMOJI)
    df["Recommendation"] = df["Risk_Level"].map(ADVICE)
 
    # -------- FILTERS --------
    st.sidebar.header("🔍 Filters")
    selected_risk = st.sidebar.multiselect("Select Risk Level", ["Low", "Medium", "High"],
                                           default=["Low", "Medium", "High"])
    prob_range = st.sidebar.slider("Churn Probability (%)", 0, 100, (0, 100))
 
    lo, hi = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
    if lo < hi:
        charge_range = st.sidebar.slider("Monthly Charges", lo, hi, (lo, hi))
    else:
        charge_range = (lo, hi)
 
    filtered_df = df[
        df["Risk_Level"].isin(selected_risk)
        & df["Churn_Probability"].between(prob_range[0], prob_range[1])
        & df["MonthlyCharges"].between(charge_range[0], charge_range[1])
    ]
 
    if filtered_df.empty:
        st.warning("No customers match the current filters. Try widening them.")
        st.stop()
 
    # -------- KPI --------
    st.subheader("📊 Key Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", len(filtered_df))
    c2.metric("High Risk", int((filtered_df["Risk_Level"] == "High").sum()))
    c3.metric("Avg Churn %", f"{filtered_df['Churn_Probability'].mean():.2f}%")
 
    # -------- TABLE --------
    st.subheader("📋 Filtered Results")
    st.dataframe(filtered_df.drop(columns=["Risk_Level"]))
 
    # -------- HIGH RISK --------
    st.subheader("🚨 High Risk Customers")
    high_df = filtered_df[filtered_df["Risk_Level"] == "High"]
    if len(high_df) > 0:
        st.error(f"{len(high_df)} customers need immediate attention!")
        st.dataframe(high_df.head(10))
    else:
        st.success("No high-risk customers 🎉")
 
    # -------- DASHBOARD --------
    st.subheader("📊 Visual Insights")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.pie(filtered_df, names="Risk_Display", hole=0.5, title="Risk Distribution"))
    with b:
        st.plotly_chart(px.histogram(filtered_df, x="Churn_Probability", title="Churn Probability (%)"))
 
    c, d = st.columns(2)
    with c:
        counts = filtered_df["Risk_Display"].value_counts().reset_index()
        counts.columns = ["Risk_Level", "Count"]
        st.plotly_chart(px.bar(counts, x="Risk_Level", y="Count", color="Risk_Level", title="Risk Count"))
    with d:
        st.plotly_chart(px.scatter(filtered_df, x="MonthlyCharges", y="Churn_Probability",
                                   color="Risk_Display", title="Charges vs Churn %"))
 
    # -------- DOWNLOAD --------
    st.download_button("📥 Download Results", filtered_df.to_csv(index=False).encode("utf-8"),
                       "results.csv", "text/csv")
 
