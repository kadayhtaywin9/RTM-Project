from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from utils.result_context import build_result_context
from utils.scenario_report import build_scenario_report, spreadsheet_safe_csv


def _report_inputs():
    frame = pd.DataFrame({
        "tower_id": [1, 2], "hazard_ai_score": [.7, .2], "hazard_type": ["cyclone"] * 2,
        "impact_status": ["coverage_scenario"] * 2, "population_affected": [100., 0.],
        "population_rerouted": [75., 0.], "coverage_impact": [25., 0.],
    })
    context = build_result_context(frame, {"mode": "local", "hazard_type": "cyclone", "ok": True}, "local", "2026-09-07T00:00:00Z")
    metrics = {"population_directly_affected": 100., "population_rerouted": 75., "population_losing_coverage": 25.}
    return frame, context, metrics


def test_download_keeps_evidence_and_scenario_assumptions():
    frame, context, metrics = _report_inputs()
    exported, report = build_scenario_report(frame, context, metrics, areas=["Yangon City"], radius_km=1.5, threshold=.5)
    assert exported.assumed_unavailable.tolist() == [True, False]
    assert exported.evidence_status.eq("demo").all()
    assert exported.scenario_service_radius_km.eq(1.5).all()
    assert exported.population_reference_year.eq(2020).all()
    assert exported.analysis_completed_at.eq("2026-09-07T00:00:00+00:00").all()
    assert report["scenario"]["unassessed_towers_assumed_available"] is True
    assert "proxy" in report["observed_outage_validation"]
    assert json.loads(json.dumps(report, allow_nan=False))["totals"]["population_losing_coverage"] == 25.
    assert "evidence_status" not in frame  # inputs are not mutated


@pytest.mark.parametrize("invalid", ["no_scenario", "duplicate", "score", "conservation", "table_total", "nan", "radius", "available_affected", "unavailable_count"])
def test_report_rejects_invalid_or_inconsistent_outputs(invalid):
    frame, context, metrics = _report_inputs()
    radius = 5.
    if invalid == "no_scenario":
        frame["impact_status"] = "requires_coverage_scenario"
    elif invalid == "duplicate":
        frame.loc[1, "tower_id"] = 1
    elif invalid == "score":
        frame.loc[0, "hazard_ai_score"] = np.inf
    elif invalid == "conservation":
        metrics["population_losing_coverage"] = 26.
    elif invalid == "table_total":
        frame.loc[0, "coverage_impact"] = 24.
    elif invalid == "nan":
        frame.loc[0, "population_affected"] = np.nan
    elif invalid == "available_affected":
        frame.loc[0, "hazard_ai_score"] = .2
    elif invalid == "unavailable_count":
        metrics["selected_failed_towers"] = 2
    else:
        radius = np.nan
    with pytest.raises(ValueError):
        build_scenario_report(frame, context, metrics, areas=["Yangon City"], radius_km=radius, threshold=.5)


def test_csv_escapes_formula_like_text_but_preserves_numbers():
    frame = pd.DataFrame({"text": ["=SUM(A1)", " +CMD", "@command", "Ordinary name"], "score": [-1., 0., 1., np.inf]})
    result = spreadsheet_safe_csv(frame).decode("utf-8-sig")
    assert "'=SUM(A1)" in result and "' +CMD" in result and "'@command" in result
    assert "Ordinary name" in result and ",-1.0" in result
    assert "inf" not in result


def test_current_radius_changes_each_export_without_changing_scores():
    frame, context, metrics = _report_inputs()
    small, _ = build_scenario_report(frame, context, metrics, areas=["Yangon City"], radius_km=1.5, threshold=.5)
    frame[["population_affected", "population_rerouted", "coverage_impact"]] = 0.
    metrics = {key: 0. for key in metrics}
    large, _ = build_scenario_report(frame, context, metrics, areas=["Yangon City"], radius_km=5., threshold=.8)
    assert small.hazard_ai_score.equals(large.hazard_ai_score)
    assert small.scenario_service_radius_km.eq(1.5).all() and large.scenario_service_radius_km.eq(5.).all()
    assert small.assumed_unavailable.tolist() == [True, False]
    assert large.assumed_unavailable.tolist() == [False, False]
