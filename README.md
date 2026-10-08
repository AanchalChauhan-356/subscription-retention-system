🚀 Subscription Retention Intelligence System

🌐 Live Demo: https://subscription-retention-system-kdbxrokufh6diw4wmyf5zu.streamlit.app/


📌 Overview

A machine learning web app that predicts which customers are likely to churn, groups them into Low, Medium and High risk, and suggests a retention action for each group. It is built for business users: upload a CSV, get predictions and insights instantly, with no code needed.

The model is trained on the IBM Telco Customer Churn dataset and deployed as an interactive Streamlit dashboard.

🎯 Objectives
Predict the probability that a customer will churn
Segment customers into Low, Medium and High risk
Highlight customers who need immediate attention
Suggest practical retention actions
Let non-technical users explore the results through a dashboard

🧠 Key Features

🔍 1. Churn Prediction
Logistic Regression model
Outputs churn probability as a percentage

🎯 2. Risk Segmentation
🟢 Low risk (<30%)
🟡 Medium risk (30–70%)
🔴 High risk (>70%)

💡 3. Recommendations
Low → Maintain engagement
Medium → Offer targeted promotions
High → Immediate retention action (discounts, support)

📊 4. Interactive Dashboard
Donut chart: risk distribution
Histogram: churn probability
Bar chart: customers per risk level
Scatter plot: monthly charges vs churn risk

🔎 5. Filters
Risk level
Churn probability (%)
Monthly charges

📥 6. CSV Upload & Download
Upload a customer CSV and get predictions instantly
Download the results as a CSV
Clear error messages if the file is missing required columns

📋 Input Format

The app expects customer data in the standard Telco format. The CSV must contain these columns (extra columns such as customerID or Churn are fine):

gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

Download sample_customers.csv from this repo to see the exact format. If a file is missing columns, the app lists which ones instead of failing.

Note: The model is trained on telecom data, so it works best on customer data with this structure. It is not a general-purpose churn model for other industries.

🏗️ Tech Stack
Area	Tools
App & deployment	Streamlit, Streamlit Cloud
Machine learning	Scikit-learn (Logistic Regression)
Data processing	Pandas, NumPy
Visualization	Plotly

📂 Project Structure
subscription-retention-system/
├── app.py                  # Streamlit application
├── model.pkl               # Trained model
├── scaler.pkl              # Feature scaler
├── columns.pkl             # Model feature columns
├── sample_customers.csv    # Example input file
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

⚙️ How It Works
The user uploads a CSV file.
The app checks that the required columns are present and cleans the data (blank or invalid numbers are filled with the median).
Categorical variables are encoded and features are scaled with the saved scaler.
The model predicts churn probability for each customer.
Probabilities are converted to percentages and grouped into risk levels with a recommendation.
The dashboard shows the results, charts and filters.

▶️ Run Locally
bash
git clone https://github.com/AanchalChauhan-356/subscription-retention-system.git
cd subscription-retention-system
pip install -r requirements.txt
streamlit run app.py

📊 Dataset

IBM Telco Customer Churn dataset (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv

🧪 Model Details
Algorithm: Logistic Regression
Why: It is easy to interpret and works well as a baseline for churn prediction.
Class imbalance: Handled with balanced class weights, since fewer customers churn than stay.
📈 Business Use

This type of system can help a business:

Spot customers who are likely to leave
Prioritise retention effort where it matters most
Act earlier instead of reacting after customers leave

🔮 Future Improvements
Column mapping so users can upload files with different column names
Stronger models (XGBoost, neural networks) and model comparison
Customer clustering for deeper segmentation
Automated email/SMS retention campaigns
Real-time API integration

👩‍💻 Author

Aanchal Chauhan BCA (AI & ML), SGT University

Built as an end-to-end project covering data preparation, modelling, deployment and business-focused insights.
