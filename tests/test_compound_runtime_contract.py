from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hazard_ai


class _CompoundModel:
    def __init__(self) -> None:
        self.inputs = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.inputs = frame.copy()
        return frame[["flood_ai_score", "earthquake_ai_score", "cyclone_ai_score"]].mean(axis=1).to_numpy()


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, _CompoundModel]:
    towers = pd.DataFrame({"tower_id": [7, 2, 9], "isolation_score": [0.1, 0.2, 0.3]})
    model = _CompoundModel()
    monkeypatch.setattr(hazard_ai, "_prepare_common", lambda frame: frame.copy())
    monkeypatch.setattr(hazard_ai, "_assets", lambda: {
        "models": {"compound": model},
        "multi_metadata": {"hazards": {"compound": {"features_in_order": [
            "flood_ai_score", "earthquake_ai_score", "cyclone_ai_score", "isolation_score",
        ]}}},
    })
    return towers, model


def _child(source: str = "Test data", timestamp: str = "2026-09-01T00:00:00+00:00") -> pd.DataFrame:
    return pd.DataFrame({
        "tower_id": [7, 2, 9],
        "hazard_ai_score": [0.1, 0.5, 0.9],
        "hazard_data_source": source,
        "hazard_data_timestamp": timestamp,
    })


def test_compound_aligns_shuffled_children_to_requested_order(runtime) -> None:
    towers, model = runtime
    flood, quake, cyclone = _child(), _child(), _child()
    quake["hazard_ai_score"] = [0.2, 0.6, 1.0]
    cyclone["hazard_ai_score"] = [0.0, 0.4, 0.8]
    result = hazard_ai._predict_compound(
        towers, flood.iloc[[2, 1, 0]], quake.iloc[[1, 0, 2]], cyclone.iloc[[1, 2, 0]], "Test compound",
    )
    assert result["tower_id"].tolist() == [7, 2, 9]
    np.testing.assert_allclose(result["hazard_ai_score"], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(model.inputs["earthquake_ai_score"], [0.2, 0.6, 1.0])
    assert result["hazard_data_source"].eq("Test compound").all()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "extra", "wrong", "missing_id", "missing_score"])
@pytest.mark.parametrize("child_index", [0, 1, 2])
def test_compound_rejects_incomplete_child_contract(runtime, invalid, child_index) -> None:
    towers, model = runtime
    children = [_child(), _child(), _child()]
    frame = children[child_index]
    if invalid == "missing":
        frame = frame.iloc[:2]
    elif invalid == "duplicate":
        frame.loc[2, "tower_id"] = 7
    elif invalid == "extra":
        frame = pd.concat([frame, frame.iloc[[0]].assign(tower_id=100)], ignore_index=True)
    elif invalid == "wrong":
        frame.loc[2, "tower_id"] = 100
    elif invalid == "missing_id":
        frame = frame.drop(columns="tower_id")
    else:
        frame = frame.drop(columns="hazard_ai_score")
    children[child_index] = frame
    with pytest.raises(ValueError, match="tower|hazard_ai_score"):
        hazard_ai._predict_compound(towers, *children, "Test compound")
    assert model.inputs is None


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf, -0.01, 1.01, "unknown", None])
@pytest.mark.parametrize("child_index", [0, 1, 2])
def test_compound_rejects_unknown_or_out_of_range_scores(runtime, bad_score, child_index) -> None:
    towers, model = runtime
    children = [_child(), _child(), _child()]
    children[child_index]["hazard_ai_score"] = pd.Series([bad_score, 0.5, 0.9], dtype=object)
    with pytest.raises(ValueError, match="scores must be finite values between 0 and 1"):
        hazard_ai._predict_compound(towers, *children, "Test compound")
    assert model.inputs is None


@pytest.mark.parametrize("ids", [[7, 7, 9], [7, None, 9], [7, 2.5, 9], [7, np.inf, 9], [7, -2, 9], [7, True, 9], []])
def test_compound_rejects_invalid_requested_ids(runtime, ids) -> None:
    _, model = runtime
    towers = pd.DataFrame({"tower_id": pd.Series(ids, dtype=object), "isolation_score": 0.2})
    with pytest.raises(ValueError, match="tower|at least one"):
        hazard_ai._predict_compound(towers, _child(), _child(), _child(), "Test compound")
    assert model.inputs is None


def test_compound_exposes_each_submodel_observation_source_and_time(runtime, monkeypatch) -> None:
    towers, _ = runtime
    monkeypatch.setattr(hazard_ai, "local_flood_predictions", lambda frame, rain_date=None: _child("Cached rainfall", "2024-12-31"))
    monkeypatch.setattr(hazard_ai, "local_earthquake_predictions", lambda frame: (
        _child("Historical seismic exposure", ""), {"source": "Packaged seismic catalog"},
    ))
    monkeypatch.setattr(hazard_ai, "local_cyclone_predictions", lambda frame: (
        _child("Historical cyclone exposure", ""), {"data_status": "background_only"},
    ))
    result, info = hazard_ai.run_hazard_ai(towers, hazard_type="compound", mode="local")
    assert info["source"] == result["hazard_data_source"].iloc[0]
    assert info["data_timestamp"] == "2024-12-31"
    assert info["timestamp_scope"] == "latest_input_only"
    assert info["submodels"]["flood"]["source"] == "Cached rainfall"
    assert info["submodels"]["flood"]["data_timestamp"] == "2024-12-31"
    assert info["submodels"]["earthquake"]["data_timestamp"] == ""
    assert info["submodels"]["earthquake"]["provider_source"] == "Packaged seismic catalog"
    assert info["submodels"]["cyclone"]["source"] == "Historical cyclone exposure"
    assert info["submodels"]["cyclone"]["data_timestamp"] == ""
    assert info["submodels"]["cyclone"]["data_status"] == "background_only"


def test_compound_auto_keeps_fallback_distinct_from_live_timestamps(runtime, monkeypatch) -> None:
    towers, _ = runtime

    def unavailable(*args, **kwargs):
        raise RuntimeError("No test credentials")

    monkeypatch.setattr(hazard_ai, "gee_flood_predictions", unavailable)
    monkeypatch.setattr(hazard_ai, "local_flood_predictions", lambda frame, rain_date=None: _child("Cached rainfall", "2024-12-31"))
    monkeypatch.setattr(hazard_ai, "live_earthquake_predictions", lambda frame: (
        _child("USGS event context", "2026-09-01T06:00:00+00:00"), {"event_count": 1},
    ))
    monkeypatch.setattr(hazard_ai, "live_cyclone_predictions", lambda frame: (
        _child("JTWC track context", "2026-09-02T00:00:00+00:00"), {"data_status": "active"},
    ))
    _, info = hazard_ai.run_hazard_ai(towers, hazard_type="compound", mode="auto")
    assert info["mode"] == "mixed"
    assert info["ok"] is False
    assert info["submodels"]["flood"]["ok"] is False
    assert info["submodels"]["flood"]["data_timestamp"] == "2024-12-31"
    assert info["submodels"]["earthquake"]["data_timestamp"] == "2026-09-01T06:00:00+00:00"
    assert info["submodels"]["cyclone"]["data_timestamp"] == "2026-09-02T00:00:00+00:00"


@pytest.mark.parametrize("missing_timestamp", ["", None, np.nan])
def test_run_info_does_not_disguise_mixed_per_tower_times(runtime, monkeypatch, missing_timestamp) -> None:
    towers, _ = runtime
    frame = _child()
    frame.loc[1, "hazard_data_timestamp"] = missing_timestamp
    monkeypatch.setattr(hazard_ai, "local_flood_predictions", lambda towers, rain_date=None: frame)
    _, info = hazard_ai.run_hazard_ai(towers, hazard_type="flood", mode="local")
    assert info["data_timestamp"] == ""
    assert info["timestamp_scope"] == "per_tower_output"
    assert info["data_timestamps"] == ["2026-09-01T00:00:00+00:00", ""]


def test_run_compound_rejects_invalid_ids_before_running_children(runtime, monkeypatch) -> None:
    towers, _ = runtime
    towers.loc[1, "tower_id"] = 7

    def should_not_run(*args, **kwargs):
        pytest.fail("Invalid requested tower IDs must be rejected before fetching child data")

    monkeypatch.setattr(hazard_ai, "gee_flood_predictions", should_not_run)
    with pytest.raises(ValueError, match="duplicate tower IDs"):
        hazard_ai.run_hazard_ai(towers, hazard_type="compound", mode="gee")
