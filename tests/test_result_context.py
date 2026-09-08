from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.decision_engine import MODEL_TYPES, DecisionEngine, _recommended_action
from utils.result_context import annotate_result_exports, build_result_context


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "tower_id": [1, 2],
        "hazard_ai_score": [0.8, 0.4],
        "hazard_data_source": ["Recorded input source"] * 2,
        "hazard_data_timestamp": ["2026-09-01T00:00:00Z"] * 2,
    })


@pytest.mark.parametrize(("requested", "actual", "ok", "expected"), [
    ("auto", "gee", True, "live"),
    ("gee", "live", True, "live"),
    ("auto", "local", False, "fallback"),
    ("auto", "local", True, "fallback"),
    ("local", "local", True, "demo"),
    ("auto", "", True, "unknown"),
    ("auto", "live", False, "unknown"),
])
def test_status_uses_recorded_mode_not_a_source_name(requested: str, actual: str, ok: bool, expected: str) -> None:
    frame = _frame()
    frame["hazard_data_source"] = "Live satellite source (cached copy)"
    context = build_result_context(frame, {"mode": actual, "ok": ok, "hazard_type": "flood"}, requested)
    assert context["evidence_status"] == expected
    assert context["completed_at"] is None
    assert "not a disaster probability" in context["score_interpretation"]
    assert context["population_reference_year"] == 2020
    assert context["source_timestamps"][0]["timestamp"] == "2026-09-01T00:00:00+00:00"
    json.dumps(context, allow_nan=False)


@pytest.mark.parametrize(("requested", "modes", "expected"), [
    ("auto", ("gee", "live", "live"), "live"),
    ("auto", ("local", "live", "live"), "mixed"),
    ("auto", ("local", "local", "local"), "fallback"),
    ("local", ("local", "local", "local"), "demo"),
    ("auto", ("gee", "live", ""), "unknown"),
])
def test_compound_evidence_and_timestamps_are_per_source(requested: str, modes: tuple[str, ...], expected: str) -> None:
    submodels = {
        kind: {"mode": mode, "ok": mode != "local" or requested == "local", "source": kind + " source", "data_timestamp": f"2026-09-0{i}T00:00:00Z"}
        for i, (kind, mode) in enumerate(zip(("flood", "earthquake", "cyclone"), modes), start=1)
    }
    context = build_result_context(_frame(), {
        "hazard_type": "compound", "mode": "live", "submodels": submodels,
        "data_timestamp": "2099-01-01T00:00:00Z",
    }, requested, "2026-09-07T00:00:00Z")
    assert context["evidence_status"] == expected
    assert len(context["sources"]) == 3
    assert "GEE" in context["sources"][0] if modes[0] == "gee" else "Packaged rainfall" in context["sources"][0]
    assert [entry["timestamp"] for entry in context["source_timestamps"]] == [
        f"2026-09-0{i}T00:00:00+00:00" for i in range(1, 4)
    ]
    assert context["completed_at"] == "2026-09-07T00:00:00+00:00"
    assert context["submodels"]["flood"]["source"] == context["sources"][0]
    assert any("separate observation times" in item for item in context["limitations"])
    exports = annotate_result_exports(_frame(), context)
    assert exports["evidence_status"].eq(expected).all()
    assert exports["flood_data_timestamps"].eq("2026-09-01T00:00:00+00:00").all()
    assert exports["cyclone_data_source"].eq(context["sources"][2]).all()
    assert "analysis_source_mode" not in _frame()
    json.dumps(context, allow_nan=False)


def test_missing_compound_child_cannot_be_reported_live() -> None:
    context = build_result_context(_frame(), {
        "hazard_type": "compound", "mode": "live", "ok": True,
        "submodels": {"flood": {"mode": "gee", "ok": True}},
    }, "auto")
    assert context["evidence_status"] == "unknown"
    assert context["submodels"]["cyclone"]["source"] == "Source not recorded"
    assert context["source_timestamps"] == []


@pytest.mark.parametrize("mode", ["auto", "local"])
def test_local_cyclone_never_implies_no_storm(mode: str) -> None:
    context = build_result_context(_frame(), {
        "hazard_type": "cyclone", "mode": "local", "ok": mode == "local", "data_status": "background_only",
    }, mode)
    assert any("Current cyclone activity is unknown" in item for item in context["limitations"])


def test_no_active_cyclone_has_source_scoped_caveat_and_separate_fetch_time() -> None:
    context = build_result_context(pd.DataFrame(), {
        "hazard_type": "cyclone", "mode": "live", "ok": True, "data_status": "no_active_storm",
        "source_checked_at": "2026-09-06T21:00:00Z",
    }, "gee", "2026-09-07T00:00:00Z")
    assert context["evidence_status"] == "live"
    assert context["source_timestamps"] == []
    assert context["source_records"][0]["source_checked_at"] == "2026-09-06T21:00:00+00:00"
    assert any("not a guarantee" in item for item in context["limitations"])


def test_bad_completion_time_or_requested_mode_rejected() -> None:
    with pytest.raises(ValueError, match="completed_at"):
        build_result_context(_frame(), {}, "auto", "not a timestamp")
    with pytest.raises(ValueError, match="requested_mode"):
        build_result_context(_frame(), {}, "invented")


def test_raw_connector_errors_and_source_urls_are_not_exported() -> None:
    context = build_result_context(_frame(), {
        "hazard_type": "flood", "mode": "local", "ok": False,
        "source": "https://service.test?token=SECRET",
        "message": "Connector error for private-project-id with SECRET credential",
        "data_status": "SECRET",
    }, "auto")
    serialized = json.dumps(context)
    assert "SECRET" not in serialized
    assert "private-project-id" not in serialized
    assert "service.test" not in serialized
    assert context["source_records"][0]["message"].startswith("Live inputs were not available")


def test_per_tower_timestamp_metadata_is_retained() -> None:
    context = build_result_context(pd.DataFrame(), {
        "hazard_type": "flood", "mode": "gee", "ok": True,
        "data_timestamps": ["", "2026-09-03T00:00:00Z", "2026-09-01T00:00:00Z"],
    }, "auto")
    assert [entry["timestamp"] for entry in context["source_timestamps"]] == [
        "2026-09-01T00:00:00+00:00", "2026-09-03T00:00:00+00:00",
    ]
    assert context["source_records"][0]["timestamp_status"] == "partly_recorded"
    assert any("partly recorded" in item for item in context["limitations"])


@pytest.mark.parametrize("missing", [None, "", np.nan, pd.NaT, "invalid timestamp"])
def test_large_frames_parse_only_distinct_timestamps_and_keep_missing_status(monkeypatch: pytest.MonkeyPatch, missing: object) -> None:
    from utils import result_context

    original = result_context._timestamp
    calls = []

    def counted_timestamp(value: object) -> str | None:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(result_context, "_timestamp", counted_timestamp)
    values = ["2026-09-01T00:00:00Z"] * 8028 + [missing]
    frame = pd.DataFrame({"hazard_data_timestamp": values})
    context = result_context.build_result_context(frame, {
        "hazard_type": "flood", "mode": "gee", "ok": True,
        "data_timestamps": values,
    }, "auto")
    assert len(calls) <= 5
    assert context["source_records"][0]["data_timestamps"] == ["2026-09-01T00:00:00+00:00"]
    assert context["source_records"][0]["timestamp_status"] == "partly_recorded"


class _ExplanationModel:
    def explanation_strings(self, frame: pd.DataFrame, top_n: int = 3) -> list[str]:
        return ["Planning factor"] * len(frame)


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf, "bad", -0.1, 1.1])
def test_invalid_scores_fail_instead_of_becoming_low_risk(monkeypatch: pytest.MonkeyPatch, invalid: object) -> None:
    monkeypatch.setitem(MODEL_TYPES, "flood", _ExplanationModel)
    frame = pd.DataFrame({"tower_id": [1], "hazard_ai_score": [invalid]})
    with pytest.raises(ValueError, match="finite numbers"):
        DecisionEngine().enrich(frame, "flood")


def test_missing_scores_rejected() -> None:
    with pytest.raises(ValueError, match="missing hazard_ai_score"):
        DecisionEngine().enrich(pd.DataFrame({"tower_id": [1]}), "flood")


def test_categories_are_consistent_with_score_even_if_input_class_is_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MODEL_TYPES, "flood", _ExplanationModel)
    frame = pd.DataFrame({"tower_id": [1, 2, 3, 4], "hazard_ai_score": [0.0, 0.3, 0.5, 0.7], "hazard_class": ["Low"] * 4})
    result = DecisionEngine().enrich(frame, "flood")
    assert result["risk_level"].tolist() == ["Low", "Moderate", "High", "Very High"]
    assert result["hazard_class"].tolist() == result["risk_level"].tolist()
    assert result["population_affected"].isna().all()


@pytest.mark.parametrize("kind", ["flood", "earthquake", "cyclone", "compound"])
@pytest.mark.parametrize("level", ["Low", "Moderate", "High", "Very High"])
def test_recommendations_are_planning_guidance_not_confirmed_response(kind: str, level: str) -> None:
    action = _recommended_action(kind, level, 15000, 0.1)
    assert action.startswith("Planning guidance:")
    assert "Activate" not in action
    assert "current" in action
    if level in {"High", "Very High"}:
        assert "before intervention" in action
    if kind == "earthquake" and level == "Very High":
        assert "only if an event is confirmed" in action


def test_hazard_engine_attaches_exportable_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine import hazard_engine

    monkeypatch.setattr(hazard_engine, "prepare_tower_features", lambda frame: frame)
    monkeypatch.setattr(hazard_engine, "_legacy_hazard_runtime", lambda *args, **kwargs: (
        _frame(), {"mode": "local", "ok": True, "hazard_type": "flood"},
    ))
    monkeypatch.setitem(MODEL_TYPES, "flood", _ExplanationModel)
    result, info = hazard_engine.HazardEngine().run(pd.DataFrame(), hazard_type="flood", mode="local")
    assert info["result_context"]["evidence_status"] == "demo"
    assert info["result_context"]["completed_at"]
    assert result["evidence_status"].eq("demo").all()
    assert result["analysis_completed_at"].eq(info["result_context"]["completed_at"]).all()
