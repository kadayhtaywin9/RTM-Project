from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hazard_ai
from data.preprocessing import prepare_tower_features
from ml import train_multi_hazard_models as training

BASE = Path(__file__).resolve().parents[1]


def test_saved_model_and_training_input_hashes_match_metadata() -> None:
    metadata = json.loads((BASE / "models" / "multi_hazard_model_metadata.json").read_text(encoding="utf-8"))
    for hazard, entry in metadata["hazards"].items():
        model_path = BASE / "models" / entry["model_file"]
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == entry["model_sha256"], hazard
    for name, expected in metadata["training_inputs"].items():
        directory = "models" if name.endswith(".json") else "data"
        assert hashlib.sha256((BASE / directory / name).read_bytes()).hexdigest() == expected, name


def test_normalized_features_do_not_depend_on_batch_selection_or_row_order() -> None:
    towers = pd.read_csv(BASE / "data" / "tower_sites_population.csv")
    full = prepare_tower_features(towers).set_index("tower_id")
    chosen = towers.iloc[[0, 50, 100, 1000, 2000, 4000, 8000]].sample(frac=1.0, random_state=42)
    features = ["tower_redundancy", "tower_vulnerability", "network_importance"]
    for subset in (chosen, chosen.iloc[:1]):
        actual = prepare_tower_features(subset).set_index("tower_id")
        pd.testing.assert_frame_equal(actual[features], full.loc[actual.index, features])


def test_training_and_runtime_redundancy_use_the_same_fixed_contract() -> None:
    towers = pd.read_csv(BASE / "data" / "tower_sites_population.csv")
    frame, normalization = training.base_frame()
    trained = frame.set_index("tower_id")
    runtime = hazard_ai._prepare_common(towers).set_index("tower_id")
    metadata = json.loads((BASE / "models" / "multi_hazard_model_metadata.json").read_text(encoding="utf-8"))
    assert normalization == metadata["feature_normalization"]
    np.testing.assert_array_equal(
        trained.loc[runtime.index, "site_redundancy_risk"].to_numpy(),
        runtime["site_redundancy_risk"].to_numpy(),
    )
    selected = towers.iloc[[0, 50, 1000, 4000]].iloc[::-1]
    subset = hazard_ai._prepare_common(selected).set_index("tower_id")
    np.testing.assert_array_equal(
        subset["site_redundancy_risk"].to_numpy(),
        runtime.loc[subset.index, "site_redundancy_risk"].to_numpy(),
    )


def test_earthquake_scores_are_identical_for_same_towers_and_event_in_any_batch() -> None:
    towers = pd.read_csv(BASE / "data" / "tower_sites_population.csv")
    full = hazard_ai._predict_earthquake(towers, np.full(len(towers), 0.6), "test").set_index("tower_id")
    chosen = towers.iloc[[0, 50, 1000, 4000]].iloc[::-1]
    for subset in (chosen, chosen.iloc[:1]):
        actual = hazard_ai._predict_earthquake(subset, np.full(len(subset), 0.6), "test").set_index("tower_id")
        np.testing.assert_array_equal(
            actual["hazard_ai_score"].to_numpy(),
            full.loc[actual.index, "hazard_ai_score"].to_numpy(),
        )
        np.testing.assert_array_equal(actual["hazard_class"].to_numpy(), full.loc[actual.index, "hazard_class"].to_numpy())


class _TrackingRegressor:
    def __init__(self, data: pd.DataFrame, features: list[str]) -> None:
        self.tower_ids = set(data["tower_id"].astype(int))
        self.townships = set(data["adm3_name"].astype(str))
        self.features = features

    def predict(self, predictors: pd.DataFrame) -> np.ndarray:
        assert predictors.columns.tolist() == self.features
        assert np.isfinite(predictors.to_numpy(float)).all()
        return np.full(len(predictors), 0.4)


def test_compound_outer_holdouts_are_never_used_by_any_upstream_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nine towers across three townships, plus an unrelated Flood-training row.
    frame = pd.DataFrame({
        "tower_id": np.arange(9),
        "adm3_name": np.repeat(["North", "Central", "South"], 3),
        "earthquake_history_score": np.linspace(0.1, 0.9, 9),
        "cyclone_history_score": np.linspace(0.9, 0.1, 9),
        "flood_history_score": np.linspace(0.2, 0.8, 9),
        "isolation_score": np.linspace(0.1, 0.9, 9),
        "elevation_risk": np.full(9, 0.3),
        "slope_risk": np.full(9, 0.4),
        "radio_vulnerability": np.full(9, 0.5),
        "site_redundancy_risk": np.full(9, 0.6),
    })
    flood_rows = frame.copy()
    flood_rows["rain_30d_percentile"] = 0.5
    flood_rows["hazard_target_score"] = 0.5
    extra = flood_rows.iloc[:1].copy()
    extra["tower_id"] = 999
    extra["adm3_name"] = "Unrelated"
    flood_rows = pd.concat([flood_rows, extra], ignore_index=True)
    _, quake_features = training.quake_training(frame)
    _, cyclone_features = training.cyclone_training(frame)
    fits = []
    dataset_calls = []
    original_dataset = training.compound_dataset

    def track_fit(data, features, params=None):
        model = _TrackingRegressor(data, features)
        fits.append(model)
        return model

    def track_dataset(base, **kwargs):
        upstream = [kwargs[name] for name in ("flood_model", "quake_model", "cyclone_model")]
        dataset_calls.append((set(base["tower_id"]), set(base["adm3_name"]), upstream))
        return original_dataset(base, **kwargs)

    monkeypatch.setattr(training, "_fit_regressor", track_fit)
    monkeypatch.setattr(training, "compound_dataset", track_dataset)
    metadata, cross_fitted = training.compound_cross_validation(
        frame,
        flood_rows,
        flood_features=hazard_ai.FLOOD_FEATURES,
        flood_params={},
        quake_features=quake_features,
        cyclone_features=cyclone_features,
    )

    assert len(fits) == 12  # Three upstream fits and one compound fit per outer fold.
    assert len(dataset_calls) == 6
    validation_tower_union = set()
    for fold_index in range(3):
        train_ids, train_townships, _ = dataset_calls[fold_index * 2]
        held_ids, held_townships, held_models = dataset_calls[fold_index * 2 + 1]
        assert not train_ids.intersection(held_ids)
        assert not train_townships.intersection(held_townships)
        assert not validation_tower_union.intersection(held_ids)
        validation_tower_union.update(held_ids)
        for upstream in held_models:
            assert upstream.tower_ids == train_ids
            assert upstream.townships == train_townships
            assert not upstream.tower_ids.intersection(held_ids)
            assert not upstream.townships.intersection(held_townships)
            assert 999 not in upstream.tower_ids
        fold = metadata["folds"][fold_index]
        assert set(fold["train_groups"]) == train_townships
        assert set(fold["validation_groups"]) == held_townships
        assert fold["upstream_validation_tower_overlap"] == 0
        assert fold["upstream_validation_group_overlap"] == 0

    assert validation_tower_union == set(frame["tower_id"])
    assert len(cross_fitted) == len(frame) * len(training.COMPOUND_SCENARIOS)
    assert not cross_fitted.duplicated(["tower_id", "compound_scenario_id"]).any()
    assert cross_fitted.groupby("tower_id").size().eq(len(training.COMPOUND_SCENARIOS)).all()
