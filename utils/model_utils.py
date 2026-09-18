"""
Explainability, confidence interpretation, and input-validation helpers.

Explainability approach: uses the model's own gain-based feature
importance (bundle["feature_importances"], computed once during training)
combined with how far each of THIS employee's values sit from the
dataset average, to produce a simple, honest "what mattered for this
prediction" view. This is not SHAP — it is a transparent approximation
built entirely from real, saved values (no fabricated numbers).
"""

import numpy as np
import pandas as pd

# Human-readable labels for the raw feature names
FEATURE_LABELS = {
    "department": "Department",
    "region": "Region",
    "education": "Education",
    "gender": "Gender",
    "recruitment_channel": "Recruitment Channel",
    "no_of_trainings": "Number of Trainings",
    "age": "Age",
    "previous_year_rating": "Previous Year Rating",
    "length_of_service": "Length of Service",
    "KPIs_met >80%": "KPI Achievement (>80%)",
    "awards_won?": "Awards Won",
    "avg_training_score": "Average Training Score",
}


def confidence_band(probability: float) -> str:
    if probability >= 0.75 or probability <= 0.25:
        return "High"
    if probability >= 0.60 or probability <= 0.40:
        return "Medium"
    return "Low"


def top_factors(employee_row: dict, df_reference: pd.DataFrame, bundle: dict, top_n: int = 5):
    """Rank the features that most likely influenced this employee's
    prediction, using saved global feature importance weighted by how far
    this employee's value deviates from the dataset mean (z-score) for
    numeric features. Returns a list of dicts: feature, raw_value,
    direction, weight."""
    importances = bundle["feature_importances"]
    numeric_cols = bundle["numeric_cols"]

    BINARY_LABELS = {
        "KPIs_met >80%": {0: "No", 1: "Yes"},
        "awards_won?": {0: "No", 1: "Yes"},
    }

    scored = []
    for feat, importance in importances.items():
        if importance <= 0:
            continue
        display_value = employee_row[feat]
        if feat in BINARY_LABELS:
            display_value = BINARY_LABELS[feat].get(display_value, display_value)
        if feat in numeric_cols:
            col = df_reference[feat]
            mean, std = col.mean(), col.std() or 1.0
            z = (employee_row[feat] - mean) / std
            direction = "above average" if z > 0.15 else ("below average" if z < -0.15 else "about average")
            weight = importance * min(abs(z), 3) / 3  # cap influence of extreme outliers
        else:
            direction = "category the model tracks"
            weight = importance * 0.6  # categorical: importance only, moderated

        scored.append({
            "feature": FEATURE_LABELS.get(feat, feat),
            "raw_value": display_value,
            "direction": direction,
            "weight": weight,
        })

    scored.sort(key=lambda d: d["weight"], reverse=True)
    return scored[:top_n]


def what_if_analysis(employee_row: dict, bundle: dict, preprocess_fn) -> list:
    """Recalculate the model's probability for a small set of realistic,
    single-field changes. Every number shown here is a real model output
    from a real recalculation — never fabricated or estimated by rule."""
    import pandas as pd

    base_df = pd.DataFrame([employee_row])
    base_result_X = preprocess_fn(base_df, bundle)
    base_prob = float(bundle["model"].predict_proba(base_result_X)[0][1])

    scenarios = []
    candidate_changes = []

    if employee_row.get("KPIs_met >80%", 0) == 0:
        candidate_changes.append(("KPI achievement", "KPIs_met >80%", 1, "Achieve KPI target (>80%)"))
    if employee_row.get("awards_won?", 0) == 0:
        candidate_changes.append(("Awards won", "awards_won?", 1, "Win a performance award"))
    if employee_row.get("previous_year_rating", 5) < 5:
        candidate_changes.append((
            "Previous year rating", "previous_year_rating",
            min(employee_row.get("previous_year_rating", 3) + 1, 5),
            "Improve previous-year rating by 1 point"
        ))
    if employee_row.get("avg_training_score", 100) < 90:
        candidate_changes.append((
            "Training score", "avg_training_score",
            min(employee_row.get("avg_training_score", 60) + 15, 100),
            "Raise average training score by 15 points"
        ))
    if employee_row.get("no_of_trainings", 0) < 3:
        candidate_changes.append((
            "Trainings completed", "no_of_trainings",
            employee_row.get("no_of_trainings", 1) + 2,
            "Complete 2 more trainings"
        ))

    for label, field, new_value, description in candidate_changes:
        modified = dict(employee_row)
        modified[field] = new_value
        mod_df = pd.DataFrame([modified])
        mod_X = preprocess_fn(mod_df, bundle)
        new_prob = float(bundle["model"].predict_proba(mod_X)[0][1])
        delta = (new_prob - base_prob) * 100
        scenarios.append({
            "label": label,
            "description": description,
            "base_probability": base_prob,
            "new_probability": new_prob,
            "delta_pct_points": delta,
        })

    scenarios.sort(key=lambda s: s["delta_pct_points"], reverse=True)
    return scenarios, base_prob


def validate_employee_inputs(age: int, length_of_service: int) -> list:
    """Returns a list of human-readable warnings for implausible
    combinations. Does not block submission — just informs the user."""
    warnings = []
    if length_of_service > (age - 16):
        warnings.append(
            f"Length of service ({length_of_service} yrs) is unusually high for "
            f"an employee of age {age} — please double-check these values."
        )
    if age < 20 and length_of_service > 2:
        warnings.append("This age/service combination is unusual — please verify.")
    return warnings
