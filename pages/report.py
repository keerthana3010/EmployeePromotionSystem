from datetime import datetime

import streamlit as st
import pandas as pd

from utils.data import load_dataset
from utils.preprocessing import load_pipeline_bundle, predict_single, preprocess_input
from utils.model_utils import confidence_band, top_factors
from utils.recommendations import generate_recommendations

st.set_page_config(page_title="Employee Report", page_icon="📄", layout="wide")

st.title("📄 Employee Promotion Report")
st.write("Generate a full AI-based promotion report for an employee, including "
         "explainability and model-based what-if recommendations.")

try:
    df = load_dataset()
    bundle = load_pipeline_bundle()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

# Temporary employee number for selection (dataset has no real employee ID)
df = df.copy()
df["Employee Number"] = range(1, len(df) + 1)

# --------------------------------------------------
# SELECT EMPLOYEE
# --------------------------------------------------

st.subheader("👤 Select Employee")

employee_number = st.selectbox("Select Employee", df["Employee Number"])
employee = df[df["Employee Number"] == employee_number].iloc[0]

employee_row = {
    "department": employee["department"],
    "region": employee["region"],
    "education": employee["education"],
    "gender": employee["gender"],
    "recruitment_channel": employee["recruitment_channel"],
    "no_of_trainings": int(employee["no_of_trainings"]),
    "age": int(employee["age"]),
    "previous_year_rating": float(employee["previous_year_rating"]),
    "length_of_service": int(employee["length_of_service"]),
    "KPIs_met >80%": int(employee["KPIs_met >80%"]),
    "awards_won?": int(employee["awards_won?"]),
    "avg_training_score": int(employee["avg_training_score"]),
}

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
    st.write(f"**Length of Service:** {employee['length_of_service']} years")
    st.write(f"**Trainings:** {employee['no_of_trainings']}")
    st.write(f"**Training Score:** {employee['avg_training_score']}")

st.divider()

# --------------------------------------------------
# PERFORMANCE INDICATORS
# --------------------------------------------------

st.subheader("📊 Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Previous Rating", employee["previous_year_rating"])
col2.metric("KPI >80%", "Yes" if employee["KPIs_met >80%"] == 1 else "No")
col3.metric("Award Won", "Yes" if employee["awards_won?"] == 1 else "No")
col4.metric("Actual Status", "Promoted" if employee["is_promoted"] == 1 else "Not Promoted")

st.divider()

# --------------------------------------------------
# ML PREDICTION (uses the saved pipeline - no runtime refitting)
# --------------------------------------------------

st.subheader("🔮 ML Promotion Prediction")

result = predict_single(pd.DataFrame([employee_row]), bundle)

if not result["success"]:
    st.error(f"⚠️ Could not generate a prediction: {result['error']}")
    st.stop()

probability = result["probability"]
prediction = result["prediction"]
confidence = confidence_band(probability)

if prediction == 1:
    st.success("✅ Employee is predicted to be PROMOTED.")
else:
    st.error("❌ Employee is predicted NOT to be promoted.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Promotion Probability", f"{probability * 100:.2f}%")
with col2:
    st.metric("Prediction Confidence", confidence)

st.caption(
    "This is a model estimate based on historical patterns, not a certainty and not a "
    "recommendation to act on its own — use it to support, not replace, human judgment."
)

st.divider()

# --------------------------------------------------
# EXPLAINABILITY
# --------------------------------------------------

st.subheader("🧠 Key Factors Behind This Prediction")

factors = top_factors(employee_row, df, bundle, top_n=5)
for f in factors:
    st.write(f"- **{f['feature']}**: {f['raw_value']} ({f['direction']})")

st.divider()

# --------------------------------------------------
# RECOMMENDATIONS (model-connected, what-if based)
# --------------------------------------------------

st.subheader("💡 Personalized Recommendations")

base_prob, recommendations = generate_recommendations(employee_row, bundle, preprocess_input)

for rec in recommendations:
    if rec["delta"] > 0:
        st.info(f"**{rec['title']}**\n\n{rec['message']}")
    else:
        st.success(rec["message"])

st.divider()

# --------------------------------------------------
# MODEL INFO
# --------------------------------------------------

st.subheader("🧪 Model Information")
mi1, mi2, mi3 = st.columns(3)
mi1.write(f"**Model:** {bundle['model_name'].replace('_', ' ').title()}")
mi2.write(f"**Decision Threshold:** {bundle['threshold']:.2f}")
mi3.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.divider()

# --------------------------------------------------
# CREATE TEXT REPORT (existing download functionality preserved)
# --------------------------------------------------

recommendation_lines = "\n".join(
    f"- {rec['title']}: {rec['message']}" for rec in recommendations
)
factor_lines = "\n".join(
    f"- {f['feature']}: {f['raw_value']} ({f['direction']})" for f in factors
)

report = f"""
EMPLOYEE PROMOTION REPORT
=========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

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
Model: {bundle['model_name'].replace('_', ' ').title()}
Decision Threshold: {bundle['threshold']:.2f}
Prediction: {"PROMOTED" if prediction == 1 else "NOT PROMOTED"}
Promotion Probability: {probability * 100:.2f}%
Prediction Confidence: {confidence}

KEY FACTORS
-----------
{factor_lines}

RECOMMENDATIONS (model-based what-if estimates)
-------------------------------------------------
{recommendation_lines}

Note: This report reflects model estimates based on historical data. It is
intended to support, not replace, human HR judgment.
"""

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

st.subheader("📥 Download Report")

st.download_button(
    label="📥 Download Employee Report",
    data=report,
    file_name=f"employee_report_{employee_number}.txt",
    mime="text/plain",
)
