import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Promotion Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Promotion Analytics")
st.write(
    "Explore employee promotion patterns and performance factors."
)

st.divider()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("final_employee_promotion_dataset.csv")

# Fill missing values
df["education"] = df["education"].fillna(
    df["education"].mode()[0]
)

df["previous_year_rating"] = df["previous_year_rating"].fillna(
    df["previous_year_rating"].median()
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total = len(df)
promoted = int(df["is_promoted"].sum())
not_promoted = total - promoted

col1, col2, col3 = st.columns(3)

col1.metric(
    "👥 Total Employees",
    f"{total:,}"
)

col2.metric(
    "🎉 Promoted",
    f"{promoted:,}"
)

col3.metric(
    "❌ Not Promoted",
    f"{not_promoted:,}"
)

st.divider()

# --------------------------------------------------
# PROMOTION BY DEPARTMENT
# --------------------------------------------------

st.subheader("🏢 Promotion by Department")

department_data = (
    df.groupby("department")["is_promoted"]
    .sum()
    .reset_index()
)

department_data.columns = [
    "Department",
    "Promoted Employees"
]

fig1 = px.bar(
    department_data,
    x="Department",
    y="Promoted Employees",
    text="Promoted Employees",
    title="Promoted Employees by Department"
)

fig1.update_layout(height=450)

st.plotly_chart(
    fig1,
    width="stretch"
)

st.divider()

# --------------------------------------------------
# PROMOTION BY KPI
# --------------------------------------------------

st.subheader("🎯 KPI Achievement vs Promotion")

kpi_data = (
    df.groupby("KPIs_met >80%")["is_promoted"]
    .mean()
    .reset_index()
)

kpi_data["KPIs_met >80%"] = kpi_data["KPIs_met >80%"].map({
    0: "KPI Not Met",
    1: "KPI Met"
})

kpi_data["Promotion Rate"] = (
    kpi_data["is_promoted"] * 100
)

fig2 = px.bar(
    kpi_data,
    x="KPIs_met >80%",
    y="Promotion Rate",
    text="Promotion Rate",
    title="Promotion Rate Based on KPI Achievement"
)

fig2.update_traces(
    texttemplate="%{text:.2f}%"
)

fig2.update_layout(
    height=450,
    yaxis_title="Promotion Rate (%)"
)

st.plotly_chart(
    fig2,
    width="stretch"
)

st.divider()

# --------------------------------------------------
# AWARDS VS PROMOTION
# --------------------------------------------------

st.subheader("🏆 Awards Won vs Promotion")

award_data = (
    df.groupby("awards_won?")["is_promoted"]
    .mean()
    .reset_index()
)

award_data["awards_won?"] = award_data["awards_won?"].map({
    0: "No Award",
    1: "Award Won"
})

award_data["Promotion Rate"] = (
    award_data["is_promoted"] * 100
)

fig3 = px.bar(
    award_data,
    x="awards_won?",
    y="Promotion Rate",
    text="Promotion Rate",
    title="Promotion Rate Based on Awards"
)

fig3.update_traces(
    texttemplate="%{text:.2f}%"
)

fig3.update_layout(
    height=450,
    yaxis_title="Promotion Rate (%)"
)

st.plotly_chart(
    fig3,
    width="stretch"
)

st.divider()

# --------------------------------------------------
# PREVIOUS YEAR RATING
# --------------------------------------------------

st.subheader("⭐ Previous Year Rating vs Promotion")

rating_data = (
    df.groupby("previous_year_rating")["is_promoted"]
    .mean()
    .reset_index()
)

rating_data["Promotion Rate"] = (
    rating_data["is_promoted"] * 100
)

fig4 = px.line(
    rating_data,
    x="previous_year_rating",
    y="Promotion Rate",
    markers=True,
    title="Promotion Rate by Previous Year Rating"
)

fig4.update_layout(
    height=450,
    xaxis_title="Previous Year Rating",
    yaxis_title="Promotion Rate (%)"
)

st.plotly_chart(
    fig4,
    width="stretch"
)

st.divider()

# --------------------------------------------------
# TRAINING SCORE
# --------------------------------------------------

st.subheader("📚 Training Score vs Promotion")

fig5 = px.box(
    df,
    x="is_promoted",
    y="avg_training_score",
    title="Training Score Distribution by Promotion Status",
    labels={
        "is_promoted": "Promotion Status",
        "avg_training_score": "Average Training Score"
    }
)

fig5.update_layout(height=450)

st.plotly_chart(
    fig5,
    width="stretch"
)

st.divider()

# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------

st.subheader("📌 Key Insights")

st.write(
    """
• Promotion is influenced by multiple employee performance factors.

• KPI achievement is an important indicator of promotion.

• Previous-year performance rating is associated with promotion outcomes.

• Awards and training performance provide additional information
  for assessing promotion eligibility.

• The XGBoost model uses these employee characteristics together
  to generate individual promotion predictions.
"""
)