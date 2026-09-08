"""Convert model scores into consistent operational planning outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import CompoundModel, CycloneModel, EarthquakeModel, FloodModel

MODEL_TYPES = {
    "flood": FloodModel,
    "cyclone": CycloneModel,
    "earthquake": EarthquakeModel,
    "compound": CompoundModel,
}


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    values = frame[column] if column in frame else pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce").fillna(default)


def attach_coverage_scenario(frame: pd.DataFrame, tower_load: pd.DataFrame) -> pd.DataFrame:
    """Attach scenario people counts, attributed to each pixel's original tower.

    Hazard scores alone do not estimate affected people. These counts come from
    the coverage simulation at the user's selected radius and failure threshold.
    """
    source_columns = {
        "baseline_people_within_radius": "baseline_population_served",
        "population_directly_affected": "population_affected",
        "population_rerouted": "population_rerouted",
        "population_losing_coverage": "coverage_impact",
    }
    required = {"tower_id", *source_columns}
    missing = required.difference(tower_load.columns)
    if missing or tower_load["tower_id"].duplicated().any():
        raise ValueError(f"Invalid coverage-scenario tower data; missing columns: {sorted(missing)}")
    lookup = tower_load.set_index("tower_id")
    out = frame.copy()
    if not out["tower_id"].isin(lookup.index).all():
        raise ValueError("Coverage scenario does not include every assessed tower")
    for source, destination in source_columns.items():
        values = pd.to_numeric(out["tower_id"].map(lookup[source]), errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all() or values.lt(0).any():
            raise ValueError(f"Invalid population counts in coverage scenario: {source}")
        out[destination] = values
    if not np.allclose(out["population_affected"], out["population_rerouted"] + out["coverage_impact"]):
        raise ValueError("Scenario affected population must equal rerouted plus losing coverage")
    out["impact_status"] = "coverage_scenario"
    redundancy = _numeric_series(out, "tower_redundancy", 0.5).clip(0, 1)
    out["recommended_action"] = [
        _recommended_action(str(kind), str(level), float(pop), float(red))
        for kind, level, pop, red in zip(
            out["hazard_type"], out["risk_level"], out["baseline_population_served"], redundancy
        )
    ]
    return out


def _recommended_action(hazard_type: str, risk_level: str, population: float, redundancy: float) -> str:
    if risk_level == "Very High":
        prefix = {
            "flood": "Prioritize an engineering review of drainage and backup-power protection",
            "cyclone": "Prioritize an engineering review of mast loading and backup-power protection",
            "earthquake": "Prioritize a structural and backhaul/power review; arrange a post-event inspection only if an event is confirmed",
            "compound": "Prioritize a multi-hazard engineering review and assess redundant service options",
        }[hazard_type]
        prefix += "; confirm current alerts and site conditions before intervention"
    elif risk_level == "High":
        prefix = "Plan an engineering review of power, backhaul, and spares; confirm current alerts and site conditions before intervention"
    elif risk_level == "Moderate":
        prefix = "Review current official hazard notices and remote alarms; verify any reported threat before intervention"
    else:
        prefix = "Continue routine monitoring and check current official notices; a low planning score does not establish site safety"
    if population >= 10000 or redundancy < 0.35:
        prefix += "; review service criticality because estimated population or limited redundancy may increase consequences"
    return "Planning guidance: " + prefix + "."


class DecisionEngine:
    """Attach risk, impact, action, and native TreeSHAP explanations."""

    def enrich(self, frame: pd.DataFrame, hazard_type: str) -> pd.DataFrame:
        if hazard_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported hazard type: {hazard_type}")
        out = frame.copy()
        out["hazard_type"] = hazard_type
        if "hazard_ai_score" not in out:
            raise ValueError("Hazard output is missing hazard_ai_score; cannot create planning decisions")
        score = pd.to_numeric(out["hazard_ai_score"], errors="coerce")
        if not np.isfinite(score.to_numpy(float)).all() or not score.between(0, 1).all():
            raise ValueError("Hazard scores must be finite numbers between 0 and 1; missing or invalid scores cannot be treated as low exposure")
        out["hazard_ai_score"] = score
        out["risk_level"] = pd.cut(
            score, bins=[-np.inf, 0.30, 0.50, 0.70, np.inf],
            labels=["Low", "Moderate", "High", "Very High"], right=False,
        ).astype(str)
        out["hazard_class"] = out["risk_level"]
        population_column = "estimated_population_primary_5km" if "estimated_population_primary_5km" in out else "estimated_population_nearest"
        population = _numeric_series(out, population_column).clip(lower=0.0)
        # Preserve the output field names but leave them unknown until the
        # explicit coverage scenario supplies people counts.
        out["population_affected"] = np.nan
        out["coverage_impact"] = np.nan
        out["population_rerouted"] = np.nan
        out["impact_status"] = "requires_coverage_scenario"
        redundancy = (
            _numeric_series(out, "tower_redundancy", 0.5)
            if "tower_redundancy" in out
            else 1.0 - _numeric_series(out, "site_redundancy_risk", 0.5)
        ).clip(0, 1)
        out["recommended_action"] = [
            _recommended_action(hazard_type, level, float(pop), float(red))
            for level, pop, red in zip(out["risk_level"], population, redundancy)
        ]
        model = MODEL_TYPES[hazard_type]()
        out["main_factors"] = model.explanation_strings(out, top_n=3)
        score_column = f"{hazard_type}_score"
        out[score_column] = score
        return out
