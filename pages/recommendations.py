import streamlit as st
import pandas as pd

from utils.data import load_dataset, get_dropdown_options
from utils.preprocessing import load_pipeline_bundle, preprocess_input
from utils.recommendations import generate_recommendations

st.set_page_config(page_title="Employee Recommendations", page_icon="💡", layout="wide")

st.title("💡 Employee Promotion Recommendations")
st.write(
    "Enter employee details to get personalized recommendations generated from the "
    "trained model's own predictions — not a fixed rule list."
)
st.divider()

try:
    df = load_dataset()
    options = get_dropdown_options(df)
    bundle = load_pipeline_bundle()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

st.subheader("Employee Details")

with st.form("recommendation_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        department = st.selectbox("Department", options["department"])
    with c2:
        region = st.selectbox("Region", options["region"])
    with c3:
        education = st.selectbox("Education", options["education"])
    with c4:
        gender = st.selectbox("Gender", options["gender"])

    recruitment_channel = st.selectbox("Recruitment Channel", options["recruitment_channel"])

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        length_of_service = st.number_input("Length of Service", min_value=1, max_value=40, value=5)
        kpi = st.selectbox("KPI Achievement >80%", ["No", "Yes"])
        awards = st.selectbox("Awards Won", ["No", "Yes"])
    with col2:
        previous_rating = st.number_input(
            "Previous Year Rating", min_value=1.0, max_value=5.0, value=3.0, step=1.0
        )
        training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=60)
        trainings = st.number_input("Number of Trainings", min_value=1, max_value=20, value=1)

    submitted = st.form_submit_button("💡 Generate Recommendations", type="primary")

st.divider()

if submitted:

    employee_row = {
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": trainings,
        "age": age,
        "previous_year_rating": previous_rating,
        "length_of_service": length_of_service,
        "KPIs_met >80%": 1 if kpi == "Yes" else 0,
        "awards_won?": 1 if awards == "Yes" else 0,
        "avg_training_score": training_score,
    }

    base_prob, recommendations = generate_recommendations(employee_row, bundle, preprocess_input)

    st.subheader("Current Model Estimate")
    st.metric("Promotion Probability", f"{base_prob * 100:.1f}%")

    st.subheader("Personalized, Model-Based Recommendations")
    st.caption(
        "Each recommendation below shows the model's recalculated probability if that "
        "specific change were made — a what-if estimate, not a guarantee or a causal claim."
    )

    for rec in recommendations:
        if rec["delta"] > 0:
            st.info(f"**{rec['title']}**\n\n{rec['message']}")
        else:
            st.success(rec["message"])
