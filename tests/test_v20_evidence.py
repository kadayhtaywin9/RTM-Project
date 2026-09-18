from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

import hazard_ai
from data import earthquake_api, gee_connector
from ui.hazard_results import _exposure_map
from utils.result_context import build_result_context
from utils.sos import sos_received_time
from utils.source_evidence import product_age_hours, source_evidence_record
from utils.visualization import EXPOSURE_COLORSCALE

BASE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("age", [0, 48, 53.7, 72, 73])
def test_flood_accepts_up_to_73_hours_without_changing_rainfall_windows(age):
    now = pd.Timestamp("2026-09-15T00:00:00Z")
    frame = pd.DataFrame({"tower_id": [1], "rainfall_24h": [2.], "rainfall_72h": [5.], "rainfall_30d": [25.]})
    frame.attrs["source_evidence"] = {"unique_hourly_images": 720, "retrieved_at": "2026-09-14T00:00:00Z"}
    result = gee_connector.validate_rainfall_samples(frame, [1], now - pd.Timedelta(hours=age), now=now)
    pd.testing.assert_frame_equal(result, frame)
    assert result.attrs == frame.attrs
    assert gee_connector.GSMAP_WINDOW_HOURS == (24, 72, 720)


def test_flood_rejects_even_slightly_over_73_hours():
    frame = pd.DataFrame({"tower_id": [1], "rainfall_24h": [2.], "rainfall_72h": [5.], "rainfall_30d": [25.]})
    with pytest.raises(RuntimeError, match="maximum is 73"):
        gee_connector.validate_rainfall_samples(frame, [1], "2026-09-11T22:59:59Z", now="2026-09-15T00:00:00Z")


@pytest.mark.parametrize("value", [None, "", "invalid", "2026-09-15 12:00:00"])
def test_sos_never_invents_a_missing_or_ambiguous_receipt_time(value):
    assert sos_received_time(value) == "Receipt time unavailable"


def test_sos_receipt_uses_myanmar_date_at_midnight():
    assert sos_received_time("2026-09-14T17:30:00Z") == "15 Sep 2026 · 00:00:00 MMT"


def test_earthquake_metadata_counts_actual_records_and_retains_dates(monkeypatch):
    features = [{
        "id": f"test-{i}", "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [96.1, 16.8, 10.]},
        "properties": {"type": "earthquake", "mag": mag, "place": "Test only",
                       "time": int(pd.Timestamp(f"2026-09-{10+i:02d}T00:00Z").timestamp() * 1000)},
    } for i, mag in enumerate([3., 4., 4.5, 5.2])]
    payload = {"type": "FeatureCollection", "metadata": {"status": 200, "count": 4}, "features": features,
               "_query_evidence": {"query_start": "2026-09-10T00:00Z", "query_end": "2026-09-14T00:00Z",
                                   "retrieved_at": "2026-09-14T00:00:01Z"}}
    monkeypatch.setattr(earthquake_api, "_fetch_geojson", lambda *args: payload)
    _, meta = earthquake_api.EarthquakeClient().tower_features(pd.DataFrame({"tower_id": [1], "lat": [16.8], "lon": [96.1]}), days=4)
    evidence = meta["source_evidence"]
    assert evidence["retrieved_records"] == evidence["accepted_events"] == 4
    assert evidence["window_days"] == 4
    assert [x["magnitude"] for x in evidence["events"]] == [3., 4., 4.5, 5.2]
    assert evidence["events"][0]["time"] == "2026-09-10T00:00:00+00:00"
    assert evidence["query_start"] == payload["_query_evidence"]["query_start"]


@pytest.mark.parametrize("kind", ["earthquake", "cyclone"])
def test_zero_event_reference_is_same_model_not_rescaled_output(kind):
    towers = pd.read_csv(BASE / "data/tower_sites_population.csv").head(12)
    predict = getattr(hazard_ai, f"_predict_{kind}")
    zero = predict(towers, np.zeros(len(towers)), "test")
    event = predict(towers, np.full(len(towers), .4), "test")
    np.testing.assert_allclose(zero.hazard_ai_score, zero.background_reference_score)
    np.testing.assert_allclose(event.background_reference_score, zero.hazard_ai_score)
    np.testing.assert_allclose(event.hazard_ai_score, event.background_reference_score + event.event_score_delta)
    assets = hazard_ai._assets()
    cols = assets["multi_metadata"]["hazards"][kind]["features_in_order"]
    expected = np.clip(assets["models"][kind].predict(event[cols]), 0, 1)
    np.testing.assert_allclose(event.hazard_ai_score, expected)


def test_no_event_does_not_get_analysis_time_as_observation_time(monkeypatch):
    towers = pd.read_csv(BASE / "data/tower_sites_population.csv").head(2)
    features = pd.DataFrame({"tower_id": towers.tower_id, "event_intensity": 0., "magnitude": 0.,
                             "depth": 0., "distance_from_epicenter": np.inf})
    monkeypatch.setattr(hazard_ai, "_recent_earthquake_features", lambda x: (features, {"event_count": 0, "strongest_time": ""}))
    result, _ = hazard_ai.live_earthquake_predictions(towers)
    assert result.hazard_data_timestamp.eq("").all()


def test_source_evidence_keeps_cache_clock_and_never_exposes_extra_keys():
    info = {"mode": "gee", "ok": True, "source_evidence": {
        "retrieved_at": "2026-09-14T00:00Z", "product_time": "2026-09-12T00:00Z",
        "unique_hourly_images": 720, "aggregation_windows_hours": [24, 72, 720],
        "private_key": "DO NOT COPY", "source_url": "secret-url",
    }}
    record = source_evidence_record("flood", info)
    assert record["details"]["unique_hourly_images"] == 720
    assert "DO NOT COPY" not in json.dumps(record) and "secret-url" not in json.dumps(record)
    assert product_age_hours(record["details"], "2026-09-15T00:00Z") == 72
    fallback = source_evidence_record("flood", dict(info, mode="local", ok=False))
    assert fallback == {"hazard_type": "flood", "external_inputs_used": False, "details": {}}


def test_compound_provenance_preserves_separate_record_counts():
    info = {"hazard_type": "compound", "submodels": {
        "flood": {"mode": "gee", "source_evidence": {"unique_hourly_images": 720}},
        "earthquake": {"mode": "live", "event_count": 4, "window_days": 4},
        "cyclone": {"mode": "local"},
    }}
    context = build_result_context(pd.DataFrame(), info, "auto")
    records = {r["hazard_type"]: r["retrieval"] for r in context["source_records"]}
    assert records["flood"]["details"]["unique_hourly_images"] == 720
    assert records["earthquake"]["details"]["accepted_events"] == 4
    assert not records["cyclone"]["external_inputs_used"]


def test_continuous_map_uses_fixed_scale_without_score_changes():
    scores = np.array([.2999, .3001, .4999, .5001, .6999, .7001])
    frame = pd.DataFrame({"tower_id": range(6), "adm3_name": "Test", "hazard_ai_score": scores,
                          "risk_level": "Moderate", "coverage_impact": np.nan,
                          "assumed_unavailable": pd.Series(pd.NA, index=range(6), dtype="boolean"),
                          "lat": 16.8, "lon": 96.1})
    original = frame.copy(deep=True)
    fig = _exposure_map(frame, "earthquake", [], {}, lambda *a, **k: go.Figure(), lambda *a: None)
    trace = fig.data[-1]
    np.testing.assert_array_equal(trace.marker.color, scores * 100)
    assert trace.marker.cmin == 0 and trace.marker.cmax == 100
    assert all(b[0] - a[0] >= .2 for a, b in pairwise(EXPOSURE_COLORSCALE))
    assert "Assumed unavailable" not in trace.hovertemplate
    pd.testing.assert_frame_equal(frame, original)
    json.loads(fig.to_json())


def test_delayed_flood_source_ui_counts_images_not_tower_rows():
    code = '''
import pandas as pd
from utils.result_context import build_result_context
from ui.source_evidence import render_source_evidence
now = pd.Timestamp.now(tz="UTC")
info = {"hazard_type":"flood", "mode":"gee", "source_evidence":{
    "unique_hourly_images":720, "aggregation_windows_hours":[24,72,720], "sample_rows":8029,
    "product_time":(now-pd.Timedelta(hours=53.7)).isoformat(), "retrieved_at":now.isoformat()}}
render_source_evidence(build_result_context(pd.DataFrame(), info, "gee"))
'''
    app = AppTest.from_string(code).run()
    assert not app.exception
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Unique hourly rainfall images"] == "720"
    assert metrics["Rainfall aggregation periods"] == "3"
    assert metrics["Sampled tower rows"] == "8,029"
    assert any("Delayed rainfall product" in warning.value for warning in app.warning)


def test_all_bandwidth_map_entry_points_removed():
    code = (BASE / "app.py").read_text(encoding="utf-8")
    assert "bandwidth" not in code.lower()
    assert not list((BASE / "ui").glob("*bandwidth*"))
    assert not list((BASE / "engine").glob("*bandwidth*"))
