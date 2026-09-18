import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import load_dataset
from utils.preprocessing import load_pipeline_bundle

st.set_page_config(page_title="Promotion Analytics", page_icon="📊", layout="wide")

st.title("📊 Promotion Analytics")
st.write("Explore employee promotion patterns and performance factors.")
st.divider()

try:
    df = load_dataset()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

# --------------------------------------------------
# FILTERS
# --------------------------------------------------

st.subheader("🔍 Filters")

f1, f2, f3, f4 = st.columns(4)
with f1:
    dept_filter = st.multiselect("Department", sorted(df["department"].unique()))
with f2:
    region_filter = st.multiselect("Region", sorted(df["region"].unique()))
with f3:
    edu_filter = st.multiselect("Education", sorted(df["education"].astype(str).unique()))
with f4:
    gender_filter = st.multiselect("Gender", sorted(df["gender"].unique()))

filtered_df = df.copy()
if dept_filter:
    filtered_df = filtered_df[filtered_df["department"].isin(dept_filter)]
if region_filter:
    filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
if edu_filter:
    filtered_df = filtered_df[filtered_df["education"].astype(str).isin(edu_filter)]
if gender_filter:
    filtered_df = filtered_df[filtered_df["gender"].isin(gender_filter)]

if filtered_df.empty:
    st.warning("No employees match the selected filters. Showing the full dataset instead.")
    filtered_df = df.copy()

st.divider()

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total = len(filtered_df)
promoted = int(filtered_df["is_promoted"].sum())
not_promoted = total - promoted
promotion_rate = (promoted / total * 100) if total else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Employees", f"{total:,}")
col2.metric("🎉 Promoted", f"{promoted:,}")
col3.metric("❌ Not Promoted", f"{not_promoted:,}")
col4.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

st.divider()

# --------------------------------------------------
# PROMOTION BY DEPARTMENT  (original chart, preserved)
# --------------------------------------------------

st.subheader("🏢 Promotion by Department")

department_data = (
    filtered_df.groupby("department")["is_promoted"].sum().reset_index()
)
department_data.columns = ["Department", "Promoted Employees"]

fig1 = px.bar(
    department_data, x="Department", y="Promoted Employees",
    text="Promoted Employees", title="Promoted Employees by Department",
)
fig1.update_layout(height=450)
st.plotly_chart(fig1, width="stretch")

st.divider()

# --------------------------------------------------
# PROMOTION BY KPI  (original chart, preserved)
# --------------------------------------------------

st.subheader("🎯 KPI Achievement vs Promotion")

kpi_data = (
    filtered_df.groupby("KPIs_met >80%")["is_promoted"].mean().reset_index()
)
kpi_data["KPIs_met >80%"] = kpi_data["KPIs_met >80%"].map({0: "KPI Not Met", 1: "KPI Met"})
kpi_data["Promotion Rate"] = kpi_data["is_promoted"] * 100

fig2 = px.bar(
    kpi_data, x="KPIs_met >80%", y="Promotion Rate", text="Promotion Rate",
    title="Promotion Rate Based on KPI Achievement",
)
fig2.update_traces(texttemplate="%{text:.2f}%")
fig2.update_layout(height=450, yaxis_title="Promotion Rate (%)")
st.plotly_chart(fig2, width="stretch")

st.divider()

# --------------------------------------------------
# AWARDS VS PROMOTION  (original chart, preserved)
# --------------------------------------------------

st.subheader("🏆 Awards Won vs Promotion")

award_data = (
    filtered_df.groupby("awards_won?")["is_promoted"].mean().reset_index()
)
award_data["awards_won?"] = award_data["awards_won?"].map({0: "No Award", 1: "Award Won"})
award_data["Promotion Rate"] = award_data["is_promoted"] * 100

fig3 = px.bar(
    award_data, x="awards_won?", y="Promotion Rate", text="Promotion Rate",
    title="Promotion Rate Based on Awards",
)
fig3.update_traces(texttemplate="%{text:.2f}%")
fig3.update_layout(height=450, yaxis_title="Promotion Rate (%)")
st.plotly_chart(fig3, width="stretch")

st.divider()

# --------------------------------------------------
# PREVIOUS YEAR RATING  (original chart, preserved)
# --------------------------------------------------

st.subheader("⭐ Previous Year Rating vs Promotion")

rating_data = (
    filtered_df.groupby("previous_year_rating")["is_promoted"].mean().reset_index()
)
rating_data["Promotion Rate"] = rating_data["is_promoted"] * 100

fig4 = px.line(
    rating_data, x="previous_year_rating", y="Promotion Rate", markers=True,
    title="Promotion Rate by Previous Year Rating",
)
fig4.update_layout(height=450, xaxis_title="Previous Year Rating", yaxis_title="Promotion Rate (%)")
st.plotly_chart(fig4, width="stretch")

st.divider()

# --------------------------------------------------
# TRAINING SCORE  (original chart, preserved)
# --------------------------------------------------

st.subheader("📚 Training Score vs Promotion")

fig5 = px.box(
    filtered_df, x="is_promoted", y="avg_training_score",
    title="Training Score Distribution by Promotion Status",
    labels={"is_promoted": "Promotion Status", "avg_training_score": "Average Training Score"},
)
fig5.update_layout(height=450)
st.plotly_chart(fig5, width="stretch")

st.divider()

# --------------------------------------------------
# NEW: AGE GROUP BREAKDOWN
# --------------------------------------------------

st.subheader("🎂 Promotion Rate by Age Group")

age_bins = [18, 25, 30, 35, 40, 45, 50, 60, 100]
age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-59", "60+"]
age_df = filtered_df.copy()
age_df["age_group"] = pd.cut(age_df["age"], bins=age_bins, labels=age_labels, right=False)

age_data = age_df.groupby("age_group", observed=True)["is_promoted"].mean().reset_index()
age_data["Promotion Rate"] = age_data["is_promoted"] * 100

fig6 = px.bar(
    age_data, x="age_group", y="Promotion Rate", text="Promotion Rate",
    title="Promotion Rate by Age Group",
)
fig6.update_traces(texttemplate="%{text:.2f}%")
fig6.update_layout(height=420, xaxis_title="Age Group", yaxis_title="Promotion Rate (%)")
st.plotly_chart(fig6, width="stretch")

st.divider()

# --------------------------------------------------
# NEW: GENDER + REGION BREAKDOWN
# --------------------------------------------------

gc1, gc2 = st.columns(2)

with gc1:
    st.subheader("🚻 Promotion Rate by Gender")
    gender_data = filtered_df.groupby("gender")["is_promoted"].mean().reset_index()
    gender_data["Promotion Rate"] = gender_data["is_promoted"] * 100
    fig7 = px.bar(gender_data, x="gender", y="Promotion Rate", text="Promotion Rate")
    fig7.update_traces(texttemplate="%{text:.2f}%")
    fig7.update_layout(height=380, xaxis_title="Gender", yaxis_title="Promotion Rate (%)")
    st.plotly_chart(fig7, width="stretch")

with gc2:
    st.subheader("🌍 Top 10 Regions by Promotion Rate")
    region_data = (
        filtered_df.groupby("region")["is_promoted"]
        .agg(["mean", "count"]).reset_index()
    )
    region_data = region_data[region_data["count"] >= 30]  # avoid noisy tiny regions
    region_data["Promotion Rate"] = region_data["mean"] * 100
    region_data = region_data.sort_values("Promotion Rate", ascending=False).head(10)
    fig8 = px.bar(region_data, x="region", y="Promotion Rate", text="Promotion Rate")
    fig8.update_traces(texttemplate="%{text:.1f}%")
    fig8.update_layout(height=380, xaxis_title="Region", yaxis_title="Promotion Rate (%)")
    st.plotly_chart(fig8, width="stretch")

st.divider()

# --------------------------------------------------
# NEW: MODEL FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("🧠 What the Model Weighs Most Heavily")

try:
    bundle = load_pipeline_bundle()
    importances = bundle.get("feature_importances", {})
    if importances:
        imp_df = pd.DataFrame(
            {"Feature": list(importances.keys()), "Importance": list(importances.values())}
        ).sort_values("Importance", ascending=True)
        fig9 = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title=f"Global Feature Importance ({bundle['model_name'].replace('_', ' ').title()})",
        )
        fig9.update_layout(height=450)
        st.plotly_chart(fig9, width="stretch")
        st.caption(
            "This reflects how much each feature contributed to the trained model's "
            "decisions overall — not a guarantee about any single employee."
        )
except FileNotFoundError:
    st.info("Train the model first (`python training/train_model.py`) to see feature importance here.")

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
  to generate individual promotion predictions — see the Model Performance
  page for how well it actually performs on unseen data.
"""
)
