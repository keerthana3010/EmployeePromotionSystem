import streamlit as st

st.set_page_config(
    page_title="Employee Recommendations",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Employee Promotion Recommendations")
st.write(
    "Enter employee performance details to receive personalized "
    "recommendations for improving promotion readiness."
)

st.divider()

st.subheader("Employee Performance")

col1, col2 = st.columns(2)

with col1:

    kpi = st.selectbox(
        "KPI Achievement >80%",
        [0, 1]
    )

    awards = st.selectbox(
        "Awards Won",
        [0, 1]
    )

    previous_rating = st.number_input(
        "Previous Year Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

with col2:

    training_score = st.number_input(
        "Average Training Score",
        min_value=0,
        max_value=100,
        value=60
    )

    trainings = st.number_input(
        "Number of Trainings",
        min_value=1,
        max_value=20,
        value=1
    )

    service = st.number_input(
        "Length of Service",
        min_value=1,
        max_value=40,
        value=5
    )

st.divider()

if st.button("💡 Generate Recommendations", type="primary"):

    recommendations = []

    if kpi == 0:
        recommendations.append(
            "🎯 Improve KPI achievement and consistently meet performance targets."
        )

    if awards == 0:
        recommendations.append(
            "🏆 Work toward achieving performance awards or recognition."
        )

    if previous_rating < 4:
        recommendations.append(
            "⭐ Improve your previous-year performance rating through consistent performance."
        )

    if training_score < 70:
        recommendations.append(
            "📚 Improve your training performance and complete relevant skill-development programs."
        )

    if trainings < 2:
        recommendations.append(
            "📈 Consider completing additional relevant training programs."
        )

    if service < 3:
        recommendations.append(
            "💼 Continue building experience and demonstrating consistent performance."
        )

    st.subheader("Personalized Recommendations")

    if recommendations:

        for recommendation in recommendations:
            st.info(recommendation)

    else:

        st.success(
            "🎉 Excellent! Your current performance indicators are strong. "
            "Continue maintaining your performance."
        )