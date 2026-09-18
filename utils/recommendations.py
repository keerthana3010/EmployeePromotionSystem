"""
Model-connected recommendation engine.

Unlike the original app (where pages/recommendations.py was a completely
separate, hardcoded rule list disconnected from the ML model), every
recommendation here is generated FROM a real model prediction and ranked
by the model's own recalculated probability change (what-if analysis).

Nothing here claims causation. Every message is explicitly framed as a
model-based estimate.
"""

from utils.model_utils import what_if_analysis


def generate_recommendations(employee_row: dict, bundle: dict, preprocess_fn, top_n: int = 5):
    """
    Returns:
        base_probability: float, the model's current estimate for this employee
        recommendations: list of dicts, ranked by estimated probability impact
    """
    scenarios, base_prob = what_if_analysis(employee_row, bundle, preprocess_fn)

    recommendations = []
    for s in scenarios[:top_n]:
        if s["delta_pct_points"] <= 0:
            continue  # only surface genuinely positive model-estimated changes
        recommendations.append({
            "title": s["description"],
            "message": (
                f"Model-estimated promotion probability increases from "
                f"{s['base_probability']*100:.1f}% to {s['new_probability']*100:.1f}% "
                f"(+{s['delta_pct_points']:.1f} percentage points) if: {s['label'].lower()} improves."
            ),
            "delta": s["delta_pct_points"],
        })

    if not recommendations:
        recommendations.append({
            "title": "Strong current profile",
            "message": (
                f"Based on the model, this employee's current profile already yields an "
                f"estimated promotion probability of {base_prob*100:.1f}%, and the tested "
                f"what-if changes did not meaningfully increase it further. "
                f"Continuing consistent performance is the model-supported path forward."
            ),
            "delta": 0.0,
        })

    return base_prob, recommendations
