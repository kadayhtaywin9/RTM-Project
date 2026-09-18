"""No-event/unknown is not green; population scenarios require usable scores."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import hazard_ai
from engine import hazard_engine
from utils.live_assessment import assessment_state, unassessed_frame
from utils.result_context import annotate_result_exports, build_result_context
from utils.scenario_report import build_scenario_report

NOW = pd.Timestamp.now(tz="UTC")
KINDS = ("flood", "earthquake", "cyclone", "compound")


def live_info(kind, empty=False):
    info = {"hazard_type": kind, "mode": "live", "ok": True}
    if kind == "flood":
        info["mode"] = "gee"
        info["source_evidence"] = {
            "hourly_image_counts": {"24": 24, "72": 72, "720": 720},
            "unique_hourly_images": 720, "aggregation_windows_hours": [24, 72, 720],
            "sample_rows": 2, "requested_towers": 2,
            "product_time": (NOW - pd.Timedelta(hours=12)).isoformat(),
            "retrieved_at": NOW.isoformat(),
        }
    elif kind == "earthquake":
        count = 0 if empty else 1
        info["event_count"] = count
        info["source_evidence"] = {
            "accepted_events": count, "retrieved_records": count, "ignored_records": 0,
            "query_start": (NOW - pd.Timedelta(days=30)).isoformat(),
            "query_end": NOW.isoformat(), "retrieved_at": NOW.isoformat(),
            "window_days": 30, "minimum_magnitude": 3,
            "events": [] if empty else [{"event_id": "TEST", "magnitude": 5., "depth_km": 10.,
                                        "time": (NOW - pd.Timedelta(days=1)).isoformat(), "place": "Test only"}],
        }
    elif kind == "cyclone":
        info.update({
            "data_status": "no_active_storm" if empty else "active",
            "active_storm_count": 0 if empty else 1, "track_points": 0 if empty else 3,
            "products_retrieved": 0 if empty else 1, "source_checked_at": NOW.isoformat(),
            "index_updated_at": NOW.isoformat(), "index_max_age_hours": 72,
            "storm_time": "" if empty else NOW.isoformat(), "advisory_max_age_hours": 36,
        })
    else:
        info["submodels"] = {name: live_info(name, empty=empty and name != "flood") for name in KINDS[:3]}
    return info


def frame():
    return pd.DataFrame({
        "tower_id": [1, 2], "lat": [16.8, 16.81], "lon": [96.1, 96.11],
        "adm3_name": ["Test", "Test"], "analysis_area": ["Test", "Test"],
        "hazard_ai_score": [.9, .8], "hazard_ai_pct": [90., 80.],
        "background_reference_score": [.435, .435], "event_score_delta": [0., 0.],
        "main_factors": ["OLD EXPLANATION", "OLD EXPLANATION"],
        "recommended_action": ["OLD ACTION", "OLD ACTION"],
        "population_affected": [100., 100.], "hazard_data_timestamp": "",
    })


@pytest.mark.parametrize("kind", KINDS)
def test_usable_inputs_allow_unchanged_planning_scores(kind):
    state = assessment_state(kind, live_info(kind), "gee", now=NOW)
    assert state["score_allowed"] and state["status"] == "live_assessed"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("failure", ["unknown", "failed", "fallback"])
def test_missing_evidence_and_strict_fallback_never_become_safe(kind, failure):
    info = {"mode": "local" if failure == "fallback" else "live", "ok": failure != "failed"}
    if kind == "compound":
        info["submodels"] = {name: dict(info) for name in KINDS[:3]}
    state = assessment_state(kind, info, "gee", now=NOW)
    assert not state["score_allowed"]
    assert state["status"] == "not_assessed"
    unknown = unassessed_frame(frame(), kind, state)
    assert unknown.hazard_ai_score.isna().all()
    assert unknown.population_affected.isna().all()
    assert unknown.risk_level.eq("Not assessed").all()
    assert "background_reference_score" not in unknown
    assert "recommended_action" not in unknown
    assert "main_factors" not in unknown


@pytest.mark.parametrize("kind", KINDS)
def test_demo_retains_planning_and_population_scenario_eligibility(kind):
    info = {"mode": "local"}
    if kind == "compound":
        info["submodels"] = {name: {"mode": "local"} for name in KINDS[:3]}
    state = assessment_state(kind, info, "local", now=NOW)
    assert state["status"] == "historical" and state["score_allowed"]


@pytest.mark.parametrize("kind", ["earthquake", "cyclone"])
def test_empty_valid_source_is_no_events_not_zero_score(kind):
    state = assessment_state(kind, live_info(kind, empty=True), "gee", now=NOW)
    assert state["status"] == "no_events"
    assert not state["score_allowed"]


@pytest.mark.parametrize("field", ["hourly_image_counts", "unique_hourly_images", "sample_rows", "requested_towers", "product_time", "retrieved_at"])
def test_flood_missing_provenance_does_not_qualify_as_live(field):
    info = live_info("flood")
    info["source_evidence"].pop(field)
    assert not assessment_state("flood", info, "gee", now=NOW)["score_allowed"]


@pytest.mark.parametrize("hours", [24, 72, 720])
def test_partial_rainfall_does_not_qualify_as_live(hours):
    info = live_info("flood")
    info["source_evidence"]["hourly_image_counts"][str(hours)] = hours - 4
    assert not assessment_state("flood", info, "gee", now=NOW)["score_allowed"]


def test_completed_flood_result_expires_without_becoming_zero_risk():
    info = live_info("flood")
    info["source_evidence"]["product_time"] = (NOW - pd.Timedelta(hours=73, seconds=1)).isoformat()
    assert not assessment_state("flood", info, "gee", now=NOW)["score_allowed"]


@pytest.mark.parametrize("problem", ["missing_time", "out_of_period", "truncated", "count_mismatch"])
def test_earthquake_incomplete_catalog_is_not_assessed(problem):
    info = live_info("earthquake")
    evidence = info["source_evidence"]
    if problem == "missing_time":
        evidence["events"][0]["time"] = ""
    elif problem == "out_of_period":
        evidence["events"][0]["time"] = (NOW - pd.Timedelta(days=40)).isoformat()
    elif problem == "truncated":
        evidence["possible_truncation"] = True
    else:
        evidence["accepted_events"] = 2
    assert not assessment_state("earthquake", info, "gee", now=NOW)["score_allowed"]


def test_cyclone_no_event_requires_a_verifiable_source_index():
    info = live_info("cyclone", empty=True)
    info["index_updated_at"] = ""
    assert assessment_state("cyclone", info, "gee", now=NOW)["status"] == "not_assessed"


def test_stale_cyclone_index_is_not_no_events():
    info = live_info("cyclone", empty=True)
    info["index_updated_at"] = (NOW - pd.Timedelta(hours=73)).isoformat()
    assert assessment_state("cyclone", info, "gee", now=NOW)["status"] == "not_assessed"


@pytest.mark.parametrize("child", KINDS[:3])
def test_compound_requires_usable_input_from_every_component(child):
    info = live_info("compound")
    info["submodels"][child] = live_info(child, empty=True) if child != "flood" else {}
    state = assessment_state("compound", info, "gee", now=NOW)
    assert not state["score_allowed"]
    assert child.title() in state["reason"]


@pytest.mark.parametrize("kind", KINDS)
def test_engine_does_not_enrich_or_export_background_as_live(monkeypatch, kind):
    info = live_info(kind, empty=True) if kind != "flood" else {"mode": "gee", "ok": True, "hazard_type": kind}
    monkeypatch.setattr(hazard_engine, "prepare_tower_features", lambda towers: towers)
    monkeypatch.setattr(hazard_engine, "_legacy_hazard_runtime", lambda *a, **k: (frame(), info))
    engine = hazard_engine.HazardEngine()

    def no_decisions(*a, **k):
        pytest.fail("Unknown/no-event input must not produce a risk ranking or TreeSHAP explanation")

    monkeypatch.setattr(engine.decisions, "enrich", no_decisions)
    result, run_info = engine.run(frame(), hazard_type=kind, mode="gee")
    assert result.hazard_ai_score.isna().all()
    assert result.population_affected.isna().all()
    assert not run_info["result_context"]["assessment"]["score_allowed"]
    assert "background_reference_score" not in result


@pytest.mark.parametrize("kind", KINDS)
def test_export_rejects_impact_totals_when_assessment_is_unavailable(kind):
    info = {"mode": "unavailable", "ok": False, "hazard_type": kind}
    context = build_result_context(frame(), info, "gee")
    exported = annotate_result_exports(frame(), context)
    assert exported.hazard_ai_score.isna().all()
    assert exported.population_affected.isna().all()
    with pytest.raises(ValueError, match="without usable model inputs"):
        build_scenario_report(frame(), context, {}, areas=["Test"], radius_km=5., threshold=.5)


@pytest.mark.parametrize("kind", ["earthquake", "cyclone"])
def test_no_events_skip_inference_entirely(monkeypatch, kind):
    info = live_info(kind, empty=True)
    if kind == "earthquake":
        monkeypatch.setattr(hazard_ai, "_recent_earthquake_features", lambda towers: (pd.DataFrame(), info))
    else:
        monkeypatch.setattr(hazard_ai, "_jtwc_cyclone_signal", lambda towers: (np.zeros(len(towers)), info, pd.DataFrame()))

    def no_prediction(*a, **k):
        pytest.fail("No-event source checks must not run background inference")

    monkeypatch.setattr(hazard_ai, f"_predict_{kind}", no_prediction)
    result, _ = hazard_ai.run_hazard_ai(frame(), hazard_type=kind, mode="gee")
    assert result.hazard_ai_score.isna().all()


def test_compound_no_events_skip_meta_model(monkeypatch):
    for kind in KINDS[:3]:
        if kind == "flood":
            data = frame()
            data.attrs["source_evidence"] = copy.deepcopy(live_info(kind)["source_evidence"])
            monkeypatch.setattr(hazard_ai, "gee_flood_predictions", lambda *a, stored=data, **k: stored.copy())
        else:
            info = live_info(kind, empty=True)
            monkeypatch.setattr(hazard_ai, f"live_{kind}_predictions", lambda towers, meta=info: (frame(), meta))

    def no_prediction(*a, **k):
        pytest.fail("Compound must not impute unavailable child scores")

    monkeypatch.setattr(hazard_ai, "_predict_compound", no_prediction)
    result, info = hazard_ai.run_hazard_ai(frame(), hazard_type="compound", mode="gee")
    assert result.hazard_ai_score.isna().all()
    assert not info["assessment"]["score_allowed"]


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_ui_has_gray_map_null_impact_and_no_historical_score(kind):
    info = live_info(kind, empty=True) if kind != "flood" else {"mode": "unavailable", "ok": False, "hazard_type": kind}
    code = f'''
import pandas as pd
import plotly.graph_objects as go
from ui.hazard_results import render_hazard_results
result = pd.DataFrame({frame().to_dict(orient="list")!r})
info = {info!r}
def forbidden(*a, **k):
    raise AssertionError("No scenario may run without usable inputs")
render_hazard_results(
    result, info, {{}}, hazard_label={kind!r}, hazard_type={kind!r}, requested_mode="gee",
    selected_areas=["Test"], service_radius_km=5., all_towers=result, pop_grid=pd.DataFrame(), area_id={{"Test":1}},
    simulate_coverage=forbidden, base_map=lambda *a, **k: go.Figure(), add_track=forbidden,
    map_config={{}}, land_cover_classes={{}}, completed_at={NOW.isoformat()!r})
'''
    app = AppTest.from_string(code).run(timeout=30)
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Affected people"] == "Not assessed"
    assert metrics["Coverage loss"] == "Not assessed"
    assert "Median model score" not in metrics
    assert not app.checkbox
    chart = app.get("plotly_chart")[0]
    trace = json.loads(chart.proto.spec)["data"][-1]
    assert trace["marker"]["color"] == "#94a3b8"
    assert not trace["marker"]["showscale"]
    assert "Exposure score" not in trace["hovertemplate"]
    assert not any("Live inputs · planning model" in widget.value for widget in app.markdown)
