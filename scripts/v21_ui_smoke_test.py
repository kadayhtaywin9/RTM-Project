"""Full app regression with a fictional SOS feed; no external API submissions."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


def by_key(widgets, key):
    return next(widget for widget in widgets if widget.key == key)


def main():
    fixture = [{"id": "SOS-TEST-ONLY", "latitude": 16.8, "longitude": 96.1, "accuracy_m": 20,
                "status": "NEW", "received_at": "2026-09-14T17:30:00Z"}]
    with patch("utils.sos.fetch_sos_incidents", return_value=(fixture, None)):
        app = AppTest.from_file(str(BASE / "app.py"), default_timeout=180).run()
        assert not app.exception
        assert not any("bandwidth" in (b.label or "").lower() for b in app.button)
        assert any("15 Sep 2026 · 00:00:00 MMT" in x.value for x in app.info)
        assert any("Received" in x.value and "00:00:00 MMT" in x.value for x in app.caption)
        next(w for w in app.selectbox if w.label == "Study area").set_value("Hmawbi").run()
        assert not app.exception
        overview = next(chart for chart in app.get("plotly_chart") if "overview_gap_map" in chart.proto.id)
        traces = {x["name"]: x for x in json.loads(overview.proto.spec)["data"]}
        assert traces["Study area boundary"]["locations"] == ["Hmawbi"]
        assert traces["Existing telecom towers"]["marker"]["size"] == 5
        assert all(row[0] == "Hmawbi" for row in traces["Existing telecom towers"]["customdata"])
        by_key(app.button, "ai_manual_assess").click().run(timeout=180)
        assert not app.exception
        labels = {"flood": "Flood / Heavy Rain", "earthquake": "Earthquake", "cyclone": "Cyclone", "compound": "Compound Risk"}
        for kind, label in labels.items():
            by_key(app.radio, "hazard_model_selector").set_value(label).run()
            by_key(app.radio, "hazard_data_source_selector").set_value("Demo data")
            by_key(app.button, "run_hazard_ai").click().run(timeout=180)
            assert not app.exception
            assert by_key(app.checkbox, f"hazard_scenario_enabled_{kind}").value
            assert any(x.label == "Offline towers" for x in app.metric)
            assert any(x.label == "Affected people" for x in app.metric)
            assert any("No live external records used" in x.value for x in app.markdown)
            original = app.session_state["hazard_ai_result"].hazard_ai_score.to_numpy().copy()
            completed = app.session_state["hazard_ai_completed_at"]
            chart = next(x for x in app.get("plotly_chart") if f"hazard_exposure_map_{kind}" in x.proto.id)
            trace = json.loads(chart.proto.spec)["data"][-1]
            assert trace["marker"]["cmin"] == 0 and trace["marker"]["cmax"] == 100
            assert "Assumed unavailable" in trace["hovertemplate"]
            by_key(app.checkbox, f"hazard_scenario_enabled_{kind}").uncheck().run(timeout=180)
            assert not any(x.label == "Offline towers" for x in app.metric)
            by_key(app.checkbox, f"hazard_scenario_enabled_{kind}").check().run(timeout=180)
            assert not app.exception
            assert any(x.label == "Offline towers" for x in app.metric)
            assert app.session_state["hazard_ai_completed_at"] == completed
            np.testing.assert_array_equal(original, app.session_state["hazard_ai_result"].hazard_ai_score)
            assert any("Initially affected (scenario)" in x.value.columns for x in app.dataframe)
            by_key(app.checkbox, f"hazard_scenario_enabled_{kind}").uncheck().run()
            assert not app.exception
            assert not any("Initially affected (scenario)" in x.value.columns for x in app.dataframe)
            # Live-source fixtures exercise the actual policy/engine/UI but do
            # not call providers. They are synthetic tests, never observations.
            synthetic = app.session_state["hazard_ai_result"].copy()
            by_key(app.radio, "hazard_data_source_selector").set_value("Live only").run()
            fixtures = runpy.run_path(str(BASE / "tests/test_v21_live_assessment.py"))
            live = fixtures["live_info"](kind)
            flood_meta = live if kind == "flood" else live.get("submodels", {}).get("flood")
            if flood_meta:
                flood_meta["source_evidence"].update(sample_rows=len(synthetic), requested_towers=len(synthetic))
            with patch("hazard_ai._run_hazard_ai", return_value=(synthetic.copy(), live)):
                by_key(app.button, "run_hazard_ai").click().run(timeout=180)
            assert not app.exception
            by_key(app.checkbox, f"hazard_scenario_enabled_{kind}").check().run(timeout=180)
            assert not app.exception
            assert any(x.label == "Affected people" and x.value != "Not assessed" for x in app.metric)
            assert app.session_state["hazard_ai_run"]["result_context"]["assessment"]["score_allowed"]
            empty = fixtures["live_info"](kind, empty=True)
            if kind == "flood":
                empty["source_evidence"]["hourly_image_counts"]["24"] = 20
            with patch("hazard_ai._run_hazard_ai", return_value=(synthetic.copy(), empty)):
                by_key(app.button, "run_hazard_ai").click().run(timeout=180)
            assert not app.exception
            assert app.session_state["hazard_ai_result"].hazard_ai_score.isna().all()
            assert any(x.label == "Affected people" and x.value == "Not assessed" for x in app.metric)
            assert not any(f"hazard_exposure_map_{kind}" in x.proto.id for x in app.get("plotly_chart"))
            gray = next(x for x in app.get("plotly_chart") if f"hazard_unassessed_map_{kind}" in x.proto.id)
            assert json.loads(gray.proto.spec)["data"][-1]["marker"]["color"] == "#94a3b8"
            with patch("hazard_ai._run_hazard_ai", side_effect=RuntimeError("TEST ONLY: external source unavailable")):
                by_key(app.button, "run_hazard_ai").click().run(timeout=180)
            assert not app.exception
            assert app.session_state["hazard_ai_result"].hazard_ai_score.isna().all()
            assert any(x.label == "Coverage loss" and x.value == "Not assessed" for x in app.metric)
            assert not any(x.label == "Offline towers" for x in app.metric)
            print(f"{kind}: default population impact, demo/live gates, gray unavailable/no-event map, cleared stale scores and source failure passed")
        by_key(app.radio, "hazard_data_source_selector").set_value("Automatic").run()
        assert not app.exception
        assert not any(x.proto.label == "Results CSV" for x in app.get("download_button"))
    print("v21 full-app smoke passed: no masks, area filtering, receipt times, site checker, four hazards, scenario isolation and stale-result hiding.")


if __name__ == "__main__":
    main()
