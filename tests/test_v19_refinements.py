from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fastapi.testclient import TestClient

from sos_service import app as service

BASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "DB_PATH", tmp_path / "test-only-sos.db")
    monkeypatch.setattr(service, "API_KEY", "test-only-key")
    with TestClient(service.app) as test_client:
        yield test_client


def test_resident_clock_asset_is_local_available_and_before_map_code(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
    assert 'MMT · UTC+06:30' in page.text
    assert page.text.index('/static/sos-clock.js?v=19') < page.text.index('leaflet@1.9.4/dist/leaflet.js')
    clock = client.get("/static/sos-clock.js?v=19")
    assert clock.status_code == 200 and "javascript" in clock.headers["content-type"]
    assert "Asia/Yangon" in clock.text and "updateClock();" in clock.text
    assert "fetch(" not in clock.text and "geolocation" not in clock.text


def test_sos_submission_and_protected_responder_actions_still_work(client):
    assert client.get("/api/sos").status_code == 401
    response = client.post("/api/sos", json={"latitude": 16.8, "longitude": 96.1, "accuracy_m": 20,
                                            "client_time": "2026-09-14T17:30:00Z"})
    assert response.status_code == 201
    incident_id = response.json()["id"]
    assert response.json()["received_at"].endswith("Z")
    url = f"/api/sos/{incident_id}/status"
    assert client.patch(url, json={"status": "ACKNOWLEDGED"}).status_code == 401
    headers = {"X-API-Key": "test-only-key"}
    assert client.patch(url, json={"status": "ACKNOWLEDGED"}, headers=headers).status_code == 200
    incidents = client.get("/api/sos", headers=headers).json()["incidents"]
    assert len(incidents) == 1 and incidents[0]["status"] == "ACKNOWLEDGED"
    assert incidents[0]["latitude"] == 16.8 and incidents[0]["client_time"] == "2026-09-14T17:30:00Z"
    assert client.post("/api/sos", json={"latitude": 91, "longitude": 96}).status_code == 422


def test_small_tower_points_keep_metadata_and_sample_without_mutation():
    tree = ast.parse((BASE / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "add_towers")
    scope = {"np": np, "go": go}
    # Isolate the local renderer without executing the dashboard or polling SOS.
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), scope)  # noqa: S102
    towers = pd.DataFrame({"lat": np.linspace(16.8, 16.9, 3000), "lon": np.full(3000, 96.1),
                           "adm3_name": "Hmawbi", "radios": "LTE", "cell_count": 2, "estimated_population_nearest": 40})
    original = towers.copy(deep=True)
    figure = scope["add_towers"](go.Figure(), towers)
    other = scope["add_towers"](go.Figure(), towers)
    trace = figure.data[0]
    assert trace.name == "Existing telecom towers" and trace.marker.symbol == "circle"
    assert trace.marker.size == 5 and trace.marker.opacity == 0.62
    assert len(trace.lat) == 2500 and "Mapped cell records" in trace.hovertemplate
    assert all(row[0] == "Hmawbi" for row in trace.customdata)
    np.testing.assert_array_equal(trace.lat, other.data[0].lat)
    pd.testing.assert_frame_equal(towers, original)


def test_sos_polling_fragment_wraps_feed_not_sound():
    tree = ast.parse((BASE / "app.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    decorators = functions["render_sos_emergency_panel"].decorator_list
    assert len(decorators) == 1 and ast.unparse(decorators[0]) == "st.fragment(run_every='5s')"
    assert not functions["sos_notification_sound"].decorator_list
