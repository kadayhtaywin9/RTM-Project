"""Portable, explicit what-if reports; no inferred accuracy or confidence."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from utils.result_context import annotate_result_exports


def build_scenario_report(
    frame: pd.DataFrame,
    context: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    areas: Sequence[str],
    radius_km: float,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep scenario assumptions and evidence attached to every exported row."""
    if not np.isfinite(radius_km) or radius_km <= 0 or not 0 <= threshold <= 1:
        raise ValueError("Scenario radius and threshold must be valid finite values")
    if frame.empty or "impact_status" not in frame or not frame["impact_status"].eq("coverage_scenario").all():
        raise ValueError("A completed coverage scenario is required for export")
    if "tower_id" not in frame or frame.tower_id.isna().any() or frame.tower_id.duplicated().any():
        raise ValueError("Export requires a unique assessed tower set")
    scores = pd.to_numeric(frame["hazard_ai_score"], errors="coerce")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Export requires finite scores between zero and one")
    count_fields = ("population_directly_affected", "population_rerouted", "population_losing_coverage")
    counts = {key: float(metrics[key]) for key in count_fields}
    if not all(np.isfinite(value) and value >= 0 for value in counts.values()):
        raise ValueError("Scenario population totals must be finite and nonnegative")
    if not np.isclose(counts[count_fields[0]], counts[count_fields[1]] + counts[count_fields[2]]):
        raise ValueError("Scenario population totals do not conserve affected people")
    for column, metric in zip(("population_affected", "population_rerouted", "coverage_impact"), count_fields, strict=True):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or values.lt(0).any() or not np.isclose(values.sum(), counts[metric]):
            raise ValueError(f"Exported tower counts do not match scenario total: {column}")
    unavailable = scores.ge(threshold)
    if frame.loc[~unavailable, "population_affected"].gt(1e-7).any():
        raise ValueError("An available tower cannot have directly affected population in this scenario")
    if not np.allclose(frame.population_affected, frame.population_rerouted + frame.coverage_impact):
        raise ValueError("Each tower's affected population must equal rerouted plus losing coverage")
    if "selected_failed_towers" in metrics and int(metrics["selected_failed_towers"]) != int(unavailable.sum()):
        raise ValueError("Assumed unavailable tower count does not match the selected threshold")

    exported = annotate_result_exports(frame, context)
    exported["scenario_service_radius_km"] = float(radius_km)
    exported["scenario_unavailable_at_score"] = float(threshold)
    exported["assumed_unavailable"] = unavailable
    exported["analysis_areas"] = " | ".join(areas)
    exported["unassessed_towers_assumed_available"] = True
    exported["scenario_interpretation"] = "What-if geographic coverage; not observed outages or affected people"
    report = {
        "report_version": "GeoVision v8",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "result_context": dict(context),
        "scenario": {
            "areas": list(areas), "service_radius_km": float(radius_km),
            "tower_unavailable_at_score_0_to_1": float(threshold),
            "unassessed_towers_assumed_available": True,
            "population_reference_year": 2020,
            "interpretation": "What-if geographic coverage, not measured outages or observed affected people",
            "all_surviving_towers_eligible": True,
        },
        "totals": {key: value.item() if isinstance(value, np.generic) else value for key, value in metrics.items()},
        "assessed_tower_count": len(frame),
        "observed_outage_validation": "Not available; models use proxy training labels",
    }
    # Reject invalid JSON numbers instead of silently issuing a misleading report.
    json.dumps(report, allow_nan=False)
    return exported, report


def spreadsheet_safe_csv(frame: pd.DataFrame) -> bytes:
    """Escape formula-like external text when opened in spreadsheet software."""
    out = frame.replace([np.inf, -np.inf], np.nan).copy()
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
        )
    return out.to_csv(index=False).encode("utf-8-sig")
