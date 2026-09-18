"""
Shared, cached dataset loading for the Employee Promotion System.
All pages should import load_dataset() from here instead of calling
pd.read_csv() directly, so the file is read and cleaned exactly once
per session instead of on every page load.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "final_employee_promotion_dataset.csv"


@st.cache_data(show_spinner="Loading employee dataset...")
def load_dataset() -> pd.DataFrame:
    """Load the dataset and apply the same missing-value handling used
    across the app (mode for education, median for previous_year_rating).
    Cached for the session — the CSV is only actually read once."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Make sure "
            "'final_employee_promotion_dataset.csv' is in the project root."
        )
    df = pd.read_csv(DATA_PATH)
    df["education"] = df["education"].fillna(df["education"].mode(dropna=True)[0])
    df["previous_year_rating"] = df["previous_year_rating"].fillna(
        df["previous_year_rating"].median()
    )
    return df


@st.cache_data(show_spinner=False)
def get_dropdown_options(df: pd.DataFrame) -> dict:
    """Sorted unique values for every categorical field, used to populate
    selectboxes consistently across pages."""
    cols = ["department", "region", "education", "gender", "recruitment_channel"]
    return {col: sorted(df[col].astype(str).unique().tolist()) for col in cols}
