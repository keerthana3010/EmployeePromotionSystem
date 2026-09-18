import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(
    page_title="Employee Promotion System",
    page_icon="📈",
    layout="wide"
) 
df = pd.read_csv("final_employee_promotion_dataset.csv")

# Fill missing values
df["education"] = df["education"].fillna(df["education"].mode()[0])
df["previous_year_rating"] = df["previous_year_rating"].fillna(
    df["previous_year_rating"].median()
)
st.title("📈 Employee Promotion Prediction System")

st.write(
    "Data-driven employee promotion assessment using machine learning."
)

st.divider()

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total_employees = len(df)
promoted = int(df["is_promoted"].sum())
not_promoted = total_employees - promoted
promotion_rate = promoted / total_employees * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 Total Employees",
    f"{total_employees:,}"
)

col2.metric(
    "🎉 Promoted",
    f"{promoted:,}"
)

col3.metric(
    "❌ Not Promoted",
    f"{not_promoted:,}"
)

col4.metric(
    "📈 Promotion Rate",
    f"{promotion_rate:.2f}%"
)

st.divider()

# --------------------------------------------------
# PROMOTION DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Promotion Distribution")

promotion_data = pd.DataFrame({
    "Status": ["Not Promoted", "Promoted"],
    "Employees": [not_promoted, promoted]
})

fig = px.bar(
    promotion_data,
    x="Status",
    y="Employees",
    text="Employees",
    title="Employee Promotion Distribution"
)

fig.update_layout(
    height=450,
    showlegend=False
)

st.plotly_chart(
    fig,
    width="stretch"
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

fig2 = px.bar(
    department_data,
    x="Department",
    y="Promoted Employees",
    text="Promoted Employees",
    title="Promoted Employees by Department"
)

fig2.update_layout(
    height=450
)

st.plotly_chart(
    fig2,
    width="stretch"
)

st.divider()

# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.subheader("ℹ️ About the System")

st.write(
    """
This system uses employee performance-related information
and a trained XGBoost machine learning model to predict
employee promotion eligibility.

### Main Features

- 🔮 Employee promotion prediction
- 📈 Promotion probability
- 💡 Personalized recommendations
- 📊 Employee analytics
- 📄 Employee promotion reports
- 📥 Downloadable reports

### Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
"""
)