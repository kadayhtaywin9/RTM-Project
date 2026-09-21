"""Presentation-neutral helpers used by the Streamlit visual layer."""
from __future__ import annotations

import pandas as pd

RISK_COLORS = {
    "Low": "#38bdf8",
    "Moderate": "#fde047",
    "High": "#f97316",
    "Very High": "#ef4444",
}

# Continuous fixed scale. No abrupt class boundaries or per-selection rescaling.
EXPOSURE_COLORSCALE = [
    [0.0, "#2563eb"], [0.30, "#38bdf8"], [0.50, "#fde047"],
    [0.70, "#f97316"], [1.0, "#ef4444"],
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
