from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hazard_ai


def _towers() -> pd.DataFrame:
    return pd.read_csv(hazard_ai.DATA / "tower_sites_population.csv").head(3)


@pytest.mark.parametrize("failure", ["missing_rain", "null_rain", "partial_towers", "stale"])
def test_live_flood_rejects_invalid_rain_and_auto_falls_back(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    towers = _towers()
    frame = pd.DataFrame({
        "tower_id": towers["tower_id"], "elevation": 10.0, "slope": 2.0,
        "rainfall_24h": 5.0, "rainfall_72h": 10.0, "rainfall_30d": 100.0,
    })
    timestamp = pd.Timestamp.now(tz="UTC")
    if failure == "missing_rain":
        frame = frame.drop(columns=["rainfall_24h", "rainfall_72h", "rainfall_30d"])
    elif failure == "null_rain":
        frame["rainfall_30d"] = np.nan
    elif failure == "partial_towers":
        frame = frame.iloc[:1]
    else:
        timestamp -= pd.Timedelta(days=44)

    class _Connector:
        def __init__(self, **kwargs):
            pass

        def environmental_features(self, requested):
            return frame.copy(), timestamp.isoformat()

    monkeypatch.setattr(hazard_ai, "EarthEngineConnector", _Connector)
    with pytest.raises(RuntimeError):
        hazard_ai.run_hazard_ai(towers, mode="gee", hazard_type="flood")
    result, metadata = hazard_ai.run_hazard_ai(towers, mode="auto", hazard_type="flood")
    assert metadata["mode"] == "local"
    assert metadata["ok"] is False
    assert result["hazard_ai_score"].between(0, 1).all()
