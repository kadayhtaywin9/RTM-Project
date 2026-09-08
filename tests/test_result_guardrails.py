from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import earthquake_api
from hazard_ai import _classify
from utils.visualization import hazard_summary


def test_unknown_impact_summary_stays_unknown() -> None:
    frame = pd.DataFrame({"tower_id": [1, 2], "risk_level": ["High", "High"], "population_affected": [np.nan, np.nan]})
    result = hazard_summary(frame)
    assert result.loc[0, "tower_count"] == 2
    assert pd.isna(result.loc[0, "population_affected"])


def test_score_threshold_boundaries_match_category() -> None:
    scores = np.array([0.0, 0.299, 0.30, 0.499, 0.50, 0.699, 0.70, 1.0])
    result = _classify(pd.DataFrame({"tower_id": range(len(scores))}), scores, "test", None, "cyclone")
    assert result["hazard_class"].tolist() == ["Low", "Low", "Moderate", "Moderate", "High", "High", "Very High", "Very High"]


@pytest.mark.parametrize("feature", [
    None,
    {"type": "Feature", "properties": {}},
    {"type": "Feature", "properties": {"type": "earthquake", "mag": 5}, "geometry": {"type": "Point", "coordinates": [96, 16, None]}},
    {"type": "Feature", "properties": {"type": "earthquake", "mag": float("nan")}, "geometry": {"type": "Point", "coordinates": [96, 16, 10]}},
])
def test_malformed_event_cannot_become_zero_impact(monkeypatch: pytest.MonkeyPatch, feature: object) -> None:
    payload = {"type": "FeatureCollection", "metadata": {"status": 200, "count": 1}, "features": [feature]}
    monkeypatch.setattr(earthquake_api, "_fetch_geojson", lambda *args: payload)
    with pytest.raises(RuntimeError, match="USGS"):
        earthquake_api.EarthquakeClient().tower_features(pd.DataFrame({"tower_id": [1], "lat": [16.8], "lon": [96.1]}))
