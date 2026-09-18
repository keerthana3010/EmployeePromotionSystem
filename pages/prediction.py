import streamlit as st
import pandas as pd

from utils.data import load_dataset, get_dropdown_options
from utils.preprocessing import load_pipeline_bundle, predict_single, preprocess_input
from utils.model_utils import confidence_band, top_factors, validate_employee_inputs

st.set_page_config(page_title="Promotion Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Employee Promotion Prediction")
st.write(
    "Enter employee details to get an AI-estimated promotion likelihood, "
    "along with the factors behind the prediction."
)
st.divider()

# --------------------------------------------------
# LOAD DATA + MODEL (cached, no refitting here)
# --------------------------------------------------

try:
    df = load_dataset()
    options = get_dropdown_options(df)
    bundle = load_pipeline_bundle()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

st.caption(
    f"Model in use: **{bundle['model_name'].replace('_', ' ').title()}** · "
    f"Decision threshold: **{bundle['threshold']:.2f}** "
    "(see the Model Performance page for how this was chosen)."
)

# --------------------------------------------------
# SECTIONED INPUT FORM
# --------------------------------------------------

with st.form("prediction_form"):

    st.subheader("👤 Employee Information")
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

    st.subheader("💼 Experience")
    e1, e2 = st.columns(2)
    with e1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
    with e2:
        length_of_service = st.number_input(
            "Length of Service (years)", min_value=1, max_value=40, value=5
        )

    st.subheader("📊 Performance")
    p1, p2, p3 = st.columns(3)
    with p1:
        previous_year_rating = st.number_input(
            "Previous Year Rating", min_value=1.0, max_value=5.0, value=3.0, step=1.0
        )
    with p2:
        kpi = st.selectbox("KPI Achievement >80%", ["No", "Yes"])
    with p3:
        awards = st.selectbox("Awards Won", ["No", "Yes"])

    st.subheader("📚 Training")
    t1, t2 = st.columns(2)
    with t1:
        no_of_trainings = st.number_input("Number of Trainings", min_value=1, max_value=20, value=1)
    with t2:
        avg_training_score = st.number_input(
            "Average Training Score", min_value=0, max_value=100, value=60
        )

    submitted = st.form_submit_button("🔮 Predict Promotion", type="primary")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    warnings = validate_employee_inputs(age, length_of_service)
    for w in warnings:
        st.warning(f"⚠️ {w}")

    employee_row = {
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "KPIs_met >80%": 1 if kpi == "Yes" else 0,
        "awards_won?": 1 if awards == "Yes" else 0,
        "avg_training_score": avg_training_score,
    }

    result = predict_single(pd.DataFrame([employee_row]), bundle)

    st.divider()
    st.subheader("Prediction Result")

    if not result["success"]:
        st.error(f"⚠️ Could not generate a prediction: {result['error']}")
    else:
        probability = result["probability"]
        prediction = result["prediction"]
        band = confidence_band(probability)

        if prediction == 1:
            st.success("✅ Employee is predicted to be PROMOTED.")
        else:
            st.error("❌ Employee is predicted NOT to be promoted.")

        rc1, rc2 = st.columns(2)
        rc1.metric("Promotion Probability", f"{probability * 100:.1f}%")
        rc2.metric("Prediction Confidence", band)

        st.caption(
            "This probability is a model estimate, not a certainty. It reflects patterns "
            "learned from historical data and should support — not replace — human judgment."
        )

        st.divider()
        st.subheader("🧠 What Influenced This Prediction")

        factors = top_factors(employee_row, df, bundle, top_n=5)
        if factors:
            for f in factors:
                st.info(f"**{f['feature']}**: {f['raw_value']} ({f['direction']})")
        else:
            st.write("No dominant factors identified for this profile.")

        st.caption(
            "Factors are ranked using the model's learned feature importance combined with "
            "how this employee's values compare to the overall dataset — not a causal claim."
        )
