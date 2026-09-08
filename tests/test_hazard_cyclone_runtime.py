from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hazard_ai


def _towers() -> pd.DataFrame:
    return pd.read_csv(hazard_ai.DATA / "tower_sites_population.csv").head(3)


def test_live_cyclone_preserves_model_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    towers = _towers()
    context = pd.DataFrame(
        {
            "tower_id": towers["tower_id"],
            "distance_to_cyclone": [120.0, 150.0, 180.0],
            "wind_speed": [70.0, 65.0, 60.0],
            "pressure": [970.0, 975.0, 980.0],
            "cyclone_category": ["Category 1", "Category 1", "Tropical Storm"],
            "storm_id": ["IO012026"] * 3,
            "storm_name": ["ALPHA"] * 3,
            "forecast_hour": [24.0] * 3,
            "forecast_valid_time": ["2026-09-05T00:00:00+00:00"] * 3,
            "track_status": ["Active JTWC forecast"] * 3,
        }
    )
    monkeypatch.setattr(
        hazard_ai,
        "_jtwc_cyclone_signal",
        lambda frame: (
            np.array([0.6, 0.5, 0.4]),
            {"data_status": "active", "storm_time": "2026-09-04T00:00:00+00:00"},
            context,
        ),
    )

    result, metadata = hazard_ai.live_cyclone_predictions(towers)

    assert len(result) == len(towers)
    assert result["tower_id"].tolist() == towers["tower_id"].tolist()
    assert result["hazard_ai_score"].between(0, 1).all()
    assert result["hazard_data_source"].str.contains("JTWC").all()
    assert metadata["data_status"] == "active"


def test_auto_cyclone_falls_back_but_strict_live_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        raise RuntimeError("feed unavailable")

    monkeypatch.setattr(hazard_ai, "live_cyclone_predictions", unavailable)
    result, metadata = hazard_ai.run_hazard_ai(_towers(), mode="auto", hazard_type="cyclone")

    assert len(result) == 3
    assert metadata["mode"] == "local"
    assert metadata["ok"] is False
    assert "JTWC" in metadata["message"]

    with pytest.raises(RuntimeError, match="feed unavailable"):
        hazard_ai.run_hazard_ai(_towers(), mode="gee", hazard_type="cyclone")


def test_local_archive_is_background_not_a_current_storm() -> None:
    result, metadata = hazard_ai.local_cyclone_predictions(_towers())
    assert result["event_intensity"].eq(0).all()
    assert result["wind_speed"].isna().all()
    assert metadata["data_status"] == "background_only"
    assert metadata["track_points"] == 0
    assert result["track_status"].str.contains("unknown").all()
