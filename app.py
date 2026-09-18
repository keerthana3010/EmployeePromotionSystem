import streamlit as st
import plotly.express as px
import pandas as pd

from utils.data import load_dataset

st.set_page_config(
    page_title="Employee Promotion System",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.title("📈 Employee Promotion Intelligence")

st.write(
    "A data-driven AI system that predicts promotion eligibility, "
    "explains *why*, and generates personalized, model-backed recommendations."
)

st.markdown(
    """
<div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin: 0.5rem 0 1rem 0;">
  <div style="flex:1; min-width:180px; background:#F5F3FF; border:1px solid #DDD6FE; border-radius:10px; padding:14px;">
    <b>1. Problem</b><br><span style="font-size:0.9em;">Which employees are ready for promotion, and why?</span>
  </div>
  <div style="flex:1; min-width:180px; background:#F5F3FF; border:1px solid #DDD6FE; border-radius:10px; padding:14px;">
    <b>2. AI Analysis</b><br><span style="font-size:0.9em;">A validated XGBoost model learns patterns from 54K+ employee records.</span>
  </div>
  <div style="flex:1; min-width:180px; background:#F5F3FF; border:1px solid #DDD6FE; border-radius:10px; padding:14px;">
    <b>3. Prediction</b><br><span style="font-size:0.9em;">Promotion likelihood + confidence for any employee profile.</span>
  </div>
  <div style="flex:1; min-width:180px; background:#F5F3FF; border:1px solid #DDD6FE; border-radius:10px; padding:14px;">
    <b>4. Recommendation</b><br><span style="font-size:0.9em;">Model-based what-if analysis: what would change the outcome.</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

try:
    df = load_dataset()

    total_employees = len(df)
    promoted = int(df["is_promoted"].sum())
    not_promoted = total_employees - promoted
    promotion_rate = promoted / total_employees * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Employees", f"{total_employees:,}")
    col2.metric("🎉 Promoted", f"{promoted:,}")
    col3.metric("❌ Not Promoted", f"{not_promoted:,}")
    col4.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

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
        promotion_data, x="Status", y="Employees", text="Employees",
        title="Employee Promotion Distribution",
        color="Status", color_discrete_map={"Not Promoted": "#C4B5FD", "Promoted": "#7C3AED"},
    )
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # --------------------------------------------------
    # PROMOTION BY DEPARTMENT
    # --------------------------------------------------

    st.subheader("🏢 Promotion by Department")

    department_data = (
        df.groupby("department")["is_promoted"].sum().reset_index()
    )
    department_data.columns = ["Department", "Promoted Employees"]

    fig2 = px.bar(
        department_data, x="Department", y="Promoted Employees",
        text="Promoted Employees", title="Promoted Employees by Department",
        color_discrete_sequence=["#7C3AED"],
    )
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, width="stretch")

except FileNotFoundError as e:
    st.error(f"⚠️ {e}")

st.divider()

# --------------------------------------------------
# NAVIGATION HINT
# --------------------------------------------------

st.subheader("🧭 Explore the App")

nc1, nc2, nc3 = st.columns(3)
with nc1:
    st.page_link("pages/prediction.py", label="🔮 Predict a Promotion", icon="🔮")
    st.page_link("pages/recommendations.py", label="💡 Get Recommendations", icon="💡")
with nc2:
    st.page_link("pages/analytics.py", label="📊 Explore Analytics", icon="📊")
    st.page_link("pages/batch_prediction.py", label="📥 Batch Predict (CSV)", icon="📥")
with nc3:
    st.page_link("pages/report.py", label="📄 Full Employee Report", icon="📄")
    st.page_link("pages/model_info.py", label="🧠 Model Performance / AI Insights", icon="🧠")

st.divider()

# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.subheader("ℹ️ About the System")

st.write(
    """
This system uses employee performance-related information and a trained
XGBoost machine learning model to predict employee promotion eligibility,
explain the prediction, and generate personalized, model-backed
recommendations.

### Main Features

- 🔮 Employee promotion prediction with confidence interpretation
- 📈 Promotion probability
- 🧠 Model explainability (feature importance + per-prediction factors)
- 💡 Personalized, model-connected recommendations with what-if analysis
- 📊 Interactive employee analytics with filters
- 📥 Batch prediction from an uploaded CSV
- 📄 Employee promotion reports (downloadable)
- 🧪 Transparent model performance / evaluation page

### Technologies Used

- Python · Pandas · Scikit-learn · XGBoost · Streamlit · Plotly
"""
)
