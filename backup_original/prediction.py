import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load model and dataset
model = joblib.load("promotion_model.pkl")
df = pd.read_csv("final_employee_promotion_dataset.csv")

# Fill missing education
df["education"] = df["education"].fillna(df["education"].mode()[0])

# Categorical columns
categorical_columns = [
    "department",
    "region",
    "education",
    "gender",
    "recruitment_channel"
]

# Create encoders
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le

# Page settings
st.set_page_config(
    page_title="Promotion Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Employee Promotion Prediction")
st.write("Enter employee details to predict promotion eligibility.")

st.divider()

st.subheader("Employee Details")

col1, col2, col3 = st.columns(3)

# Column 1
with col1:

    department = st.selectbox(
        "Department",
        sorted(df["department"].astype(str).unique())
    )

    region = st.selectbox(
        "Region",
        sorted(df["region"].astype(str).unique())
    )

    education = st.selectbox(
        "Education",
        sorted(df["education"].astype(str).unique())
    )

    gender = st.selectbox(
        "Gender",
        sorted(df["gender"].astype(str).unique())
    )

# Column 2
with col2:

    recruitment_channel = st.selectbox(
        "Recruitment Channel",
        sorted(df["recruitment_channel"].astype(str).unique())
    )

    no_of_trainings = st.number_input(
        "Number of Trainings",
        min_value=1,
        max_value=20,
        value=1
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    previous_year_rating = st.number_input(
        "Previous Year Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

# Column 3
with col3:

    length_of_service = st.number_input(
        "Length of Service",
        min_value=1,
        max_value=40,
        value=5
    )

    kpi = st.selectbox(
        "KPI Achievement >80%",
        [0, 1]
    )

    awards = st.selectbox(
        "Awards Won",
        [0, 1]
    )

    avg_training_score = st.number_input(
        "Average Training Score",
        min_value=0,
        max_value=100,
        value=60
    )

st.divider()

# Prediction
if st.button("🔮 Predict Promotion", type="primary"):

    input_data = pd.DataFrame([{
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "KPIs_met >80%": kpi,
        "awards_won?": awards,
        "avg_training_score": avg_training_score
    }])

    # Encode categorical columns
    for col in categorical_columns:
        input_data[col] = encoders[col].transform(
            input_data[col].astype(str)
        )

    # Keep same feature order as model
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

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Employee is predicted to be PROMOTED.")
    else:
        st.error("❌ Employee is predicted NOT to be promoted.")

    st.metric(
        "Promotion Probability",
        f"{probability * 100:.2f}%"
    )