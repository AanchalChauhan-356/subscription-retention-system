🚀 Subscription Retention Intelligence System

📌 Overview

The Subscription Retention Intelligence System is a machine learning-powered web application that predicts customer churn, segments users into risk categories, and provides actionable recommendations to improve customer retention.

This project is built using the IBM Telco Customer Churn dataset and deployed as an interactive dashboard.

🎯 Objectives

Predict the probability of customer churn

Segment customers into Low, Medium, and High risk categories

Identify customers requiring immediate attention

Provide actionable retention strategies

Enable business users to analyze data through an interactive dashboard

🧠 Key Features

🔍 1. Churn Prediction

 Uses Logistic Regression model

 Outputs churn probability as a percentage

🎯 2. Risk Segmentation

* 🟢 Low Risk (<30%)
* 🟡 Medium Risk (30–70%)
* 🔴 High Risk (>70%)

💡 3. Smart Recommendations

Low → Maintain engagement

Medium → Offer targeted promotions

High → Immediate retention actions (discounts/support)

📊 4. Interactive Dashboard

Donut chart (risk distribution)

Histogram (churn probability)

Bar chart (risk counts)

Scatter plot (charges vs churn risk)

🔍 5. Filters

Risk Level filter

Churn Probability (%) filter

Monthly Charges filter

📥 6. CSV Upload & Download

Upload customer dataset

Get predictions instantly

Download results as CSV

🏗️ Tech Stack

Frontend & Deployment: Streamlit

Machine Learning: Scikit-learn

Data Processing: Pandas, NumPy

Visualization: Plotly

Model: Logistic Regression

📂 Project Structure


subscription-retention-system/

│
├── app.py              # Main Streamlit application

├── model.pkl           # Trained ML model

├── scaler.pkl          # Feature scaler

├── columns.pkl         # Feature columns

├── requirements.txt    # Dependencies

└── README.md           # Project documentation

⚙️ How It Works

1. User uploads a CSV file
2. Data is preprocessed:
   Missing values handled
   Categorical variables encoded
3. Data is scaled using saved scaler
4. Model predicts churn probability
5. Results are:
   Converted to percentage
   Categorized into risk levels
   Displayed with recommendations
6. Dashboard visualizes insights

▶️ Installation & Setup

1. Clone Repository

https://github.com/AanchalChauhan-356/subscription-retention-system

2. Install Dependencies

pip install -r requirements.txt

3. Run Application

https://subscription-retention-system-kdbxrokufh6diw4wmyf5zu.streamlit.app/

 🌐 Deployment

The application is deployed using Streamlit Cloud.

📊 Dataset

IBM Telco Customer Churn Dataset (Kaggle)

🧪 Model Details

 Algorithm: Logistic Regression
 Reason for selection:

  Handles imbalanced data effectively
  
  High interpretability
  
  Good recall for churn prediction

📈 Business Impact

This system helps businesses:

  Identify customers likely to churn
  
  Take proactive retention actions
  
  Improve customer lifetime value
  
  Reduce revenue loss

🔮 Future Improvements

Real-time API integration

Advanced models (XGBoost, Neural Networks)

Customer segmentation clustering

Email/SMS automation for retention

 👨‍💻 Author

Developed as part of a data science project to demonstrate end-to-end ML deployment with business insights.

Name : Aanchal Chauhan 

Course : BCA 4A AIML

University : SGT University 

⭐ Conclusion

This project showcases how machine learning can be applied to solve real-world business problems by combining prediction, analytics, and decision-making in one system.

