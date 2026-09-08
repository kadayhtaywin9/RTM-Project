from __future__ import annotations

import json
from typing import Any, Self
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from data import earthquake_api


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _catalog(features: list[dict[str, Any]], *, status: int = 200) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "metadata": {"status": status, "count": len(features)},
        "features": features,
    }


def _event(event_type: str, magnitude: float, *, lon: float = 96.1, lat: float = 16.8) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 10.0]},
        "properties": {
            "type": event_type,
            "mag": magnitude,
            "place": f"Test {event_type}",
            "time": 1_788_649_200_000,
        },
    }


def test_usgs_query_filters_event_type_and_uses_exact_window(monkeypatch: Any) -> None:
    requested_url = ""

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        nonlocal requested_url
        requested_url = request.full_url
        assert timeout == 15
        return _FakeResponse(_catalog([]))

    monkeypatch.setattr(earthquake_api, "urlopen", fake_urlopen)

    result = earthquake_api._fetch_geojson.__wrapped__((12.0, 20.0, 90.0, 100.0), 30, 3.0)

    query = parse_qs(urlparse(requested_url).query)
    start = pd.Timestamp(query["starttime"][0])
    end = pd.Timestamp(query["endtime"][0])
    assert result["features"] == []
    assert query["eventtype"] == ["earthquake"]
    assert end - start == pd.Timedelta(days=30)
    assert "T" in query["starttime"][0]
    assert "T" in query["endtime"][0]


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "upstream failed"},
        {"type": "FeatureCollection", "features": []},
        {"type": "FeatureCollection", "metadata": {"status": 200}},
        {"type": "FeatureCollection", "metadata": {"status": 503, "count": 0}, "features": []},
        {"type": "FeatureCollection", "metadata": {"status": 200, "count": 2}, "features": []},
    ],
)
def test_usgs_validator_rejects_error_or_malformed_payloads(payload: object) -> None:
    with pytest.raises(RuntimeError):
        earthquake_api.validate_usgs_geojson(payload)


def test_error_shaped_cached_payload_cannot_be_reported_as_zero_events(monkeypatch: Any) -> None:
    monkeypatch.setattr(earthquake_api, "_fetch_geojson", lambda bounds, days, minimum: {"error": "failed"})
    towers = pd.DataFrame({"tower_id": [1], "lat": [16.8], "lon": [96.1]})

    with pytest.raises(RuntimeError, match="FeatureCollection"):
        earthquake_api.EarthquakeClient().tower_features(towers)


def test_valid_empty_catalog_remains_a_zero_event_result(monkeypatch: Any) -> None:
    monkeypatch.setattr(earthquake_api, "_fetch_geojson", lambda bounds, days, minimum: _catalog([]))
    towers = pd.DataFrame({"tower_id": [1], "lat": [16.8], "lon": [96.1]})

    result, metadata = earthquake_api.EarthquakeClient().tower_features(towers)

    assert metadata["event_count"] == 0
    assert result["event_intensity"].tolist() == [0.0]


def test_non_earthquake_events_are_excluded_from_features_and_metadata(monkeypatch: Any) -> None:
    payload = _catalog(
        [
            _event("quarry blast", 7.0),
            _event("earthquake", 4.5),
        ]
    )
    monkeypatch.setattr(earthquake_api, "_fetch_geojson", lambda bounds, days, minimum: payload)
    towers = pd.DataFrame({"tower_id": [1], "lat": [16.8], "lon": [96.1]})

    result, metadata = earthquake_api.EarthquakeClient().tower_features(towers)

    assert metadata["event_count"] == 1
    assert metadata["ignored_event_count"] == 1
    assert metadata["magnitude"] == 4.5
    assert result.loc[0, "magnitude"] == 4.5
    assert 0 < result.loc[0, "event_intensity"] <= 1
