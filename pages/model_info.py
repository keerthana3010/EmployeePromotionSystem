import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_pipeline_bundle

st.set_page_config(page_title="Model Performance", page_icon="🧪", layout="wide")

st.title("🧪 Model Performance / AI Insights")
st.write(
    "A transparent look at how the deployed model was trained, validated, "
    "and how well it actually performs — no fabricated numbers."
)
st.divider()

COMPARISON_PATH = Path(__file__).resolve().parent.parent / "training" / "model_comparison.json"

if not COMPARISON_PATH.exists():
    st.error(
        "⚠️ No evaluation report found. Run `python training/train_model.py` "
        "to generate `training/model_comparison.json`."
    )
    st.stop()

report = json.loads(COMPARISON_PATH.read_text())

# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

st.subheader("📦 Dataset")

ds = report["dataset"]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Total Employees (raw)", f"{ds['total_rows_raw']:,}")
d2.metric("Duplicates Removed", f"{ds['duplicates_removed']:,}")
d3.metric("Promoted", f"{ds['promoted']:,}")
d4.metric("Not Promoted", f"{ds['not_promoted']:,}")

st.write(
    f"**Class balance:** {ds['promotion_rate_pct']}% promoted "
    f"— a rare-positive-class problem, which is why Accuracy alone is misleading "
    f"and Precision/Recall/F1/PR-AUC matter more."
)
st.write(f"**Train / Test split:** {ds['train_size']:,} / {ds['test_size']:,} "
         f"(stratified 80/20, test set never used in training or threshold tuning)")

st.divider()

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.subheader("🤖 Model Comparison")

candidates = report["candidates"]
rows = []
for name, m in candidates.items():
    rows.append({
        "Model": name.replace("_", " ").title(),
        "Accuracy": m["accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "ROC-AUC": m["roc_auc"],
        "PR-AUC": m["pr_auc"],
        "Threshold Used": m["threshold"],
    })
comparison_df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
st.dataframe(comparison_df.style.format({
    "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Threshold Used": "{:.2f}",
}), width="stretch")

st.caption(
    "All candidates evaluated on the SAME held-out test set, using a per-model threshold "
    "chosen to maximize F1 on cross-validated training data (never on the test set itself)."
)

st.write("**How class imbalance was handled:**")
for k, v in report["class_imbalance_handling"].items():
    st.write(f"- {k.replace('_', ' ').title()}: `{v}`")

st.divider()

# --------------------------------------------------
# SELECTED MODEL
# --------------------------------------------------

st.subheader(f"✅ Selected Model: {report['selected_model'].replace('_', ' ').title()}")

sm = report["selected_model_metrics"]
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Accuracy", f"{sm['accuracy']:.3f}")
s2.metric("Precision", f"{sm['precision']:.3f}")
s3.metric("Recall", f"{sm['recall']:.3f}")
s4.metric("F1", f"{sm['f1']:.3f}")
s5.metric("ROC-AUC", f"{sm['roc_auc']:.3f}")
s6.metric("PR-AUC", f"{sm['pr_auc']:.3f}")

cm = sm["confusion_matrix"]  # [[TN, FP], [FN, TP]]
cm_df = pd.DataFrame(
    cm, index=["Actual: Not Promoted", "Actual: Promoted"],
    columns=["Predicted: Not Promoted", "Predicted: Promoted"]
)
fig_cm = px.imshow(
    cm_df, text_auto=True, color_continuous_scale="Purples",
    title="Confusion Matrix (Test Set)"
)
fig_cm.update_layout(height=420)
st.plotly_chart(fig_cm, width="stretch")

st.divider()

# --------------------------------------------------
# TRANSPARENT COMPARISON TO ORIGINAL MODEL (data leakage finding)
# --------------------------------------------------

if report.get("original_model"):
    st.subheader("⚠️ Comparison to the Original (Pre-Upgrade) Model")

    om = report["original_model"]
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("Accuracy", f"{om['accuracy']:.3f}")
    o2.metric("Precision", f"{om['precision']:.3f}")
    o3.metric("Recall", f"{om['recall']:.3f}")
    o4.metric("F1", f"{om['f1']:.3f}")
    o5.metric("ROC-AUC", f"{om['roc_auc']:.3f}")
    o6.metric("PR-AUC", f"{om['pr_auc']:.3f}")

    leak = report.get("data_leakage_finding", {})
    if leak:
        st.warning(
            f"**Important finding:** {leak.get('summary', '')}\n\n"
            f"**Conclusion:** {leak.get('conclusion', '')}"
        )

st.divider()

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("🧠 Global Feature Importance")

importances = report.get("feature_importances", {})
if importances:
    imp_df = pd.DataFrame(
        {"Feature": list(importances.keys()), "Importance": list(importances.values())}
    ).sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
    fig_imp.update_layout(height=450)
    st.plotly_chart(fig_imp, width="stretch")

st.divider()

# --------------------------------------------------
# HOW IT WORKS (non-technical)
# --------------------------------------------------

st.subheader("❓ How the AI Prediction Works")

st.write(
    """
1. **Data**: The model learned from ~54,700 historical employee records
   (after removing exact duplicates), covering demographics, training,
   performance ratings, KPIs, awards, and tenure.
2. **Training**: The dataset was split into a training set and a completely
   separate test set. The model only ever saw the training set while learning.
3. **Handling rare promotions**: Only about 8.5% of employees were promoted.
   Instead of ignoring this imbalance (which would make the model just
   predict "not promoted" for everyone and still look "accurate"), the model
   was trained with class-imbalance handling so it pays proper attention to
   the minority (promoted) class.
4. **Validation**: The test set — data the model never trained on — is used
   to report every metric on this page. This is the honest, generalizable
   estimate of how the model performs on new employees.
5. **Prediction**: For a new employee, the model outputs a probability. If
   that probability crosses the tuned decision threshold, it predicts
   "Promoted."
6. **Explainability**: Feature importance shows which factors the model
   leans on most heavily across all employees; the Prediction and Report
   pages show factors specific to one employee.

This system is a decision-support tool. It estimates likelihood based on
historical patterns — it does not make promotion decisions and does not
claim that any one factor *causes* promotion.
"""
)
