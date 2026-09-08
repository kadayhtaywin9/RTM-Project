"""Presentation-neutral helpers used by the Streamlit visual layer."""
from __future__ import annotations

import pandas as pd

RISK_COLORS = {
    "Low": "#22c55e",
    "Moderate": "#f59e0b",
    "High": "#f97316",
    "Very High": "#dc2626",
}


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
