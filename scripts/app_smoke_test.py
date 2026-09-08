"""Render the Streamlit app and exercise every local hazard workspace."""
from __future__ import annotations

import json
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest


def by_key(widgets, key: str):
    return next(widget for widget in widgets if widget.key == key)


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=180).run()
    assert not app.exception
    assert any('id="university-team-header"' in str(block.value) for block in app.markdown)

    hazard_labels = {
        "flood": "Flood / Heavy Rain",
        "cyclone": "Cyclone",
        "earthquake": "Earthquake",
        "compound": "Compound Risk",
    }
    for hazard_type, hazard_label in hazard_labels.items():
        by_key(app.radio, "hazard_model_selector").set_value(hazard_label)
        app.run(timeout=180)
        by_key(app.radio, "hazard_data_source_selector").set_value("Demo data")
        by_key(app.button, "run_hazard_ai").click()
        started = time.perf_counter()
        app.run(timeout=180)
        elapsed = time.perf_counter() - started
        assert not app.exception
        scenario_tables = [table.value for table in app.dataframe if "Initially affected (scenario)" in table.value.columns]
        assert scenario_tables
        assert scenario_tables[0]["Initially affected (scenario)"].notna().all()
        assert any("Demo / historical inputs" in str(block.value) and "Research model" in str(block.value) for block in app.markdown)
        assert app.session_state["hazard_ai_run"]["result_context"]["evidence_status"] == "demo"
        assert app.session_state["hazard_ai_result"]["recommended_action"].str.startswith("Planning guidance:").all()
        exposure_traces = [trace for chart in app.get("plotly_chart") for trace in json.loads(chart.proto.spec)["data"] if trace.get("name") == "Tower exposure"]
        assert len(exposure_traces) == 1
        assert exposure_traces[0]["marker"]["cmax"] == 100
        assert ".1%" not in exposure_traces[0]["hovertemplate"]
        download_labels = [item.proto.label for item in app.get("download_button")]
        assert "Download tower results (CSV)" in download_labels
        assert "Download scenario notes (JSON)" in download_labels
        print(f"{hazard_type} workspace rendered in {elapsed:.2f}s")

    # Scenario controls recalculate population impact without rerunning inference.
    completed_at = app.session_state["hazard_ai_completed_at"]
    next(widget for widget in app.slider if widget.label == "Population service radius").set_value(1.5)
    app.run(timeout=180)
    assert not app.exception
    assert app.session_state["hazard_ai_completed_at"] == completed_at
    assert any("People served before scenario (1.5 km)" in table.value.columns for table in app.dataframe)

    # Selecting Live only must not leave the previous demo results on screen.
    by_key(app.radio, "hazard_data_source_selector").set_value("Live only")
    app.run(timeout=180)
    assert not app.exception
    assert any("Analysis settings changed" in item.value for item in app.warning)
    assert not any("Initially affected (scenario)" in table.value.columns for table in app.dataframe)
    assert not any(item.proto.label == "Download tower results (CSV)" for item in app.get("download_button"))

    print("Streamlit app smoke test passed.")


if __name__ == "__main__":
    main()
