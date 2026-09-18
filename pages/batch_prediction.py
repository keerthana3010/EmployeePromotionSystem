import streamlit as st
import pandas as pd

from utils.preprocessing import load_pipeline_bundle, predict_batch, REQUIRED_BATCH_COLUMNS

st.set_page_config(page_title="Batch Prediction", page_icon="📥", layout="wide")

st.title("📥 Batch Promotion Prediction")
st.write(
    "Upload a CSV of multiple employees to get promotion predictions for all of them at once."
)

with st.expander("📋 Required CSV columns"):
    st.code(", ".join(REQUIRED_BATCH_COLUMNS))
    st.write(
        "Extra columns are fine and will be kept in the output. "
        "Missing `education` / `previous_year_rating` values are automatically "
        "handled the same way as during training."
    )

st.divider()

try:
    bundle = load_pipeline_bundle()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload employee CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Could not read this file as a CSV: {e}")
        st.stop()

    missing_cols = [c for c in REQUIRED_BATCH_COLUMNS if c not in batch_df.columns]
    if missing_cols:
        st.error(
            "⚠️ This file is missing required columns:\n\n"
            + "\n".join(f"- `{c}`" for c in missing_cols)
        )
        st.stop()

    st.success(f"✅ File loaded: {len(batch_df):,} rows.")
    st.write("Preview:")
    st.dataframe(batch_df.head(10), width="stretch")

    if st.button("🔮 Run Batch Prediction", type="primary"):
        with st.spinner("Running predictions..."):
            try:
                # Drop rows that are entirely unusable (all required fields missing)
                usable_mask = batch_df[REQUIRED_BATCH_COLUMNS].isnull().all(axis=1) == False
                usable_df = batch_df[usable_mask].copy()
                skipped = len(batch_df) - len(usable_df)

                results_df = predict_batch(usable_df, bundle)

                if skipped:
                    st.warning(f"⚠️ Skipped {skipped} row(s) with no usable data.")

                st.subheader("Results")
                st.dataframe(results_df, width="stretch")

                summary1, summary2, summary3 = st.columns(3)
                promoted_count = (results_df["predicted_promotion"] == "Promoted").sum()
                summary1.metric("Total Predicted", f"{len(results_df):,}")
                summary2.metric("Predicted Promoted", f"{promoted_count:,}")
                summary3.metric(
                    "Predicted Promotion Rate",
                    f"{promoted_count / len(results_df) * 100:.2f}%" if len(results_df) else "0%",
                )

                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name="batch_promotion_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(
                    f"⚠️ Batch prediction failed: {e}\n\n"
                    "Please check that the uploaded values match the expected formats "
                    "(e.g. numeric columns should not contain text)."
                )
