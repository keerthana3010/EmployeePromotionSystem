import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Report",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Employee Promotion Report")
st.write("Generate an ML-based promotion report for an employee.")

# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

model = joblib.load("promotion_model.pkl")

df = pd.read_csv("final_employee_promotion_dataset.csv")

# Fill missing values
df["education"] = df["education"].fillna(
    df["education"].mode()[0]
)

df["previous_year_rating"] = df["previous_year_rating"].fillna(
    df["previous_year_rating"].median()
)

# Create temporary employee number
df["Employee Number"] = range(1, len(df) + 1)

# --------------------------------------------------
# CREATE LABEL ENCODERS
# --------------------------------------------------

categorical_columns = [
    "department",
    "region",
    "education",
    "gender",
    "recruitment_channel"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le

# --------------------------------------------------
# SELECT EMPLOYEE
# --------------------------------------------------

st.subheader("👤 Select Employee")

employee_number = st.selectbox(
    "Select Employee",
    df["Employee Number"]
)

employee = df[
    df["Employee Number"] == employee_number
].iloc[0]

st.divider()

# --------------------------------------------------
# EMPLOYEE DETAILS
# --------------------------------------------------

st.subheader("👤 Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**Employee Number:** {employee_number}")
    st.write(f"**Department:** {employee['department']}")
    st.write(f"**Region:** {employee['region']}")

with col2:
    st.write(f"**Education:** {employee['education']}")
    st.write(f"**Gender:** {employee['gender']}")
    st.write(f"**Age:** {employee['age']}")

with col3:
    st.write(
        f"**Length of Service:** "
        f"{employee['length_of_service']} years"
    )

    st.write(
        f"**Trainings:** "
        f"{employee['no_of_trainings']}"
    )

    st.write(
        f"**Training Score:** "
        f"{employee['avg_training_score']}"
    )

st.divider()

# --------------------------------------------------
# PERFORMANCE INDICATORS
# --------------------------------------------------

st.subheader("📊 Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Previous Rating",
    employee["previous_year_rating"]
)

col2.metric(
    "KPI >80%",
    "Yes" if employee["KPIs_met >80%"] == 1 else "No"
)

col3.metric(
    "Award Won",
    "Yes" if employee["awards_won?"] == 1 else "No"
)

col4.metric(
    "Actual Status",
    "Promoted" if employee["is_promoted"] == 1
    else "Not Promoted"
)

st.divider()

# --------------------------------------------------
# ML PREDICTION
# --------------------------------------------------

st.subheader("🔮 ML Promotion Prediction")

# Create input data
input_data = pd.DataFrame([{
    "department": employee["department"],
    "region": employee["region"],
    "education": employee["education"],
    "gender": employee["gender"],
    "recruitment_channel": employee["recruitment_channel"],
    "no_of_trainings": employee["no_of_trainings"],
    "age": employee["age"],
    "previous_year_rating": employee["previous_year_rating"],
    "length_of_service": employee["length_of_service"],
    "KPIs_met >80%": employee["KPIs_met >80%"],
    "awards_won?": employee["awards_won?"],
    "avg_training_score": employee["avg_training_score"]
}])

# Encode categorical columns
for col in categorical_columns:
    input_data[col] = encoders[col].transform(
        input_data[col].astype(str)
    )

# Keep same feature order as prediction.py
input_data = input_data[
    [
        "department",
        "region",
        "education",
        "gender",
        "recruitment_channel",
        "no_of_trainings",
        "age",
        "previous_year_rating",
        "length_of_service",
        "KPIs_met >80%",
        "awards_won?",
        "avg_training_score"
    ]
]

# Prediction
prediction = model.predict(input_data)

# Probability
probability = model.predict_proba(input_data)[0][1]

# Confidence
if probability >= 0.75:
    confidence = "High"
elif probability >= 0.50:
    confidence = "Medium"
else:
    confidence = "Low"

# --------------------------------------------------
# DISPLAY PREDICTION
# --------------------------------------------------

if prediction[0] == 1:
    st.success(
        "✅ Employee is predicted to be PROMOTED."
    )
else:
    st.error(
        "❌ Employee is predicted NOT to be promoted."
    )

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Promotion Probability",
        f"{probability * 100:.2f}%"
    )

with col2:
    st.metric(
        "Prediction Confidence",
        confidence
    )

st.divider()

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

st.subheader("💡 Personalized Recommendations")

recommendations = []

if employee["KPIs_met >80%"] == 0:
    recommendations.append(
        "🎯 Improve KPI achievement and consistently meet performance targets."
    )

if employee["awards_won?"] == 0:
    recommendations.append(
        "🏆 Work toward achieving performance awards or recognition."
    )

if employee["previous_year_rating"] < 4:
    recommendations.append(
        "⭐ Improve your previous-year performance rating."
    )

if employee["avg_training_score"] < 70:
    recommendations.append(
        "📚 Improve training performance and complete relevant skill-development programs."
    )

if employee["no_of_trainings"] < 2:
    recommendations.append(
        "📈 Consider completing additional relevant training programs."
    )

if employee["length_of_service"] < 3:
    recommendations.append(
        "💼 Continue building experience and demonstrating consistent performance."
    )

if not recommendations:
    recommendations.append(
        "🌟 Maintain your current performance and continue demonstrating consistent results."
    )

for recommendation in recommendations:
    st.info(recommendation)

st.divider()

# --------------------------------------------------
# CREATE TEXT REPORT
# --------------------------------------------------

report = f"""
EMPLOYEE PROMOTION REPORT
=========================

Employee Number: {employee_number}

EMPLOYEE DETAILS
----------------
Department: {employee['department']}
Region: {employee['region']}
Education: {employee['education']}
Gender: {employee['gender']}
Age: {employee['age']}
Length of Service: {employee['length_of_service']} years
Number of Trainings: {employee['no_of_trainings']}
Average Training Score: {employee['avg_training_score']}
Previous Year Rating: {employee['previous_year_rating']}

PERFORMANCE
-----------
KPI >80%: {"Yes" if employee["KPIs_met >80%"] == 1 else "No"}
Award Won: {"Yes" if employee["awards_won?"] == 1 else "No"}

ML PREDICTION
-------------
Prediction: {"PROMOTED" if prediction[0] == 1 else "NOT PROMOTED"}
Promotion Probability: {probability * 100:.2f}%
Prediction Confidence: {confidence}

RECOMMENDATIONS
---------------
"""

for recommendation in recommendations:
    report += recommendation + "\n"

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

st.subheader("📥 Download Report")

st.download_button(
    label="📥 Download Employee Report",
    data=report,
    file_name=f"employee_report_{employee_number}.txt",
    mime="text/plain"
)