"""
Loads the SAVED preprocessing + model bundle produced by
training/train_model.py.

This is the fix for the biggest reliability issue in the original app:
encoders used to be refit from the live CSV on every single page load,
which is fragile (breaks if the dataset changes) and crashes on unseen
categories. Now the encoder, imputers, feature order, model and decision
threshold are all fit ONCE during training and loaded here as-is.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PIPELINE_PATH = Path(__file__).resolve().parent.parent / "models" / "promotion_pipeline.joblib"


@st.cache_resource(show_spinner="Loading trained model...")
def load_pipeline_bundle() -> dict:
    """Load the trained model + fitted encoder + imputers + metadata.
    Cached as a resource so it's loaded once per app process, not per page."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found at {PIPELINE_PATH}. "
            "Run `python training/train_model.py` first to generate it."
        )
    return joblib.load(PIPELINE_PATH)


def preprocess_input(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply the exact same imputation + encoding used during training to
    new employee data. Unseen categories are safely mapped to -1 by the
    saved OrdinalEncoder (handle_unknown='use_encoded_value') instead of
    raising an exception.
    """
    df = raw_df.copy()

    impute_values = bundle["impute_values"]
    if "education" in df.columns:
        df["education"] = df["education"].fillna(impute_values["education_mode"])
    if "previous_year_rating" in df.columns:
        df["previous_year_rating"] = df["previous_year_rating"].fillna(
            impute_values["rating_median"]
        )

    encoder = bundle["encoder"]
    cat_cols = bundle["categorical_cols"]
    df[cat_cols] = encoder.transform(df[cat_cols].astype(str))

    return df[bundle["feature_order"]]


def predict_single(raw_df: pd.DataFrame, bundle: dict) -> dict:
    """Run a full prediction for one employee row. Returns prediction,
    probability, and the threshold used — never raises on bad input,
    returns an error message instead so the UI can show something useful."""
    try:
        X = preprocess_input(raw_df, bundle)
        model = bundle["model"]
        probability = float(model.predict_proba(X)[0][1])
        threshold = bundle["threshold"]
        prediction = int(probability >= threshold)
        return {
            "success": True,
            "prediction": prediction,
            "probability": probability,
            "threshold": threshold,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_batch(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Run predictions for a dataframe of many employees at once.
    Rows that fail preprocessing (e.g. missing required columns) are
    reported as errors rather than crashing the whole batch."""
    X = preprocess_input(raw_df, bundle)
    model = bundle["model"]
    threshold = bundle["threshold"]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = raw_df.copy()
    result["predicted_promotion"] = ["Promoted" if p == 1 else "Not Promoted" for p in preds]
    result["promotion_probability_pct"] = (probs * 100).round(2)
    return result


REQUIRED_BATCH_COLUMNS = [
    "department", "region", "education", "gender", "recruitment_channel",
    "no_of_trainings", "age", "previous_year_rating", "length_of_service",
    "KPIs_met >80%", "awards_won?", "avg_training_score",
]
