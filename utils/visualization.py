"""Presentation-neutral helpers used by the Streamlit visual layer."""
from __future__ import annotations

import pandas as pd

RISK_COLORS = {
    "Low": "#77a992",
    "Moderate": "#c8bb83",
    "High": "#cd9777",
    "Very High": "#b96570",
}

# Continuous fixed scale. No abrupt class boundaries or per-selection rescaling.
EXPOSURE_COLORSCALE = [
    [0.0, "#77a992"], [0.25, "#a6b58c"], [0.5, "#d3be88"],
    [0.75, "#cd9777"], [1.0, "#b96570"],
]


def hazard_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable risk-category summary suitable for charts or tables."""
    if frame.empty or "risk_level" not in frame:
        return pd.DataFrame(columns=["risk_level", "tower_count", "population_affected"])
    return (
        frame.groupby("risk_level", observed=False, as_index=False)
        .agg(
            tower_count=("tower_id", "count"),
            population_affected=("population_affected", lambda values: values.sum(min_count=1)),
        )
    )
