"""Train GeoVision earthquake, cyclone, and compound proxy models.

The targets in this module are transparent MVP planning formulas. They are not
observed disaster, tower-outage, or service-impact labels and must not be
presented as calibrated physical probabilities.

The compound stack uses the exact packaged Flood feature contract and
geographically cross-fitted upstream predictions. During outer validation, no
upstream model is fitted on a held-out tower or township.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = BASE / "models"

RANDOM_STATE = 42
INTENSITIES = np.array([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], dtype=float)
POPULATION_COLUMN = "estimated_population_primary_5km"
COMPOUND_FEATURES = ["flood_ai_score", "earthquake_ai_score", "cyclone_ai_score", "isolation_score"]

# Six deterministic proxy scenarios retain the prior row count while exposing
# each upstream model to single- and multi-hazard conditions.
COMPOUND_SCENARIOS = (
    {"rain_30d_percentile": 0.0, "earthquake_event_intensity": 0.0, "cyclone_event_intensity": 0.0},
    {"rain_30d_percentile": 0.2, "earthquake_event_intensity": 0.8, "cyclone_event_intensity": 0.2},
    {"rain_30d_percentile": 0.4, "earthquake_event_intensity": 0.2, "cyclone_event_intensity": 0.8},
    {"rain_30d_percentile": 0.6, "earthquake_event_intensity": 0.6, "cyclone_event_intensity": 0.4},
    {"rain_30d_percentile": 0.8, "earthquake_event_intensity": 0.4, "cyclone_event_intensity": 0.6},
    {"rain_30d_percentile": 1.0, "earthquake_event_intensity": 1.0, "cyclone_event_intensity": 1.0},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_optional(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    values = frame[name] if name in frame.columns else pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(values, errors="coerce").fillna(default).astype(float)


def feature_normalization(frame: pd.DataFrame) -> dict[str, Any]:
    """Create fixed normalizers from the complete packaged training universe."""
    cell_count_max = float(_numeric_optional(frame, "cell_count", 1.0).clip(lower=1.0).max())
    population_max = float(_numeric_optional(frame, POPULATION_COLUMN, 0.0).clip(lower=0.0).max())
    if cell_count_max < 1.0 or population_max <= 0.0:
        raise ValueError("Training data cannot produce valid fixed feature normalizers")
    return {
        "schema_version": 1,
        "cell_count_max": cell_count_max,
        "cell_count_transform": "log1p(clip(value, 1, cell_count_max)) / log1p(cell_count_max)",
        "population_column": POPULATION_COLUMN,
        "population_max": population_max,
        "population_transform": "log1p(clip(value, 0, population_max)) / log1p(population_max)",
        "scope": "complete packaged 8,029-tower training universe; never recompute per request",
    }


def radio_vulnerability(series: pd.Series) -> np.ndarray:
    text = series.fillna("").astype(str).str.upper()
    technology_count = text.str.count(",") + 1
    has_lte = text.str.contains("LTE").astype(float)
    has_umts = text.str.contains("UMTS").astype(float)
    value = 0.74 - 0.12 * (technology_count - 1) - 0.10 * has_lte - 0.03 * has_umts
    return np.clip(value.to_numpy(float), 0.25, 0.80)


def base_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    towers = pd.read_csv(DATA / "tower_sites_population.csv")
    static = pd.read_csv(DATA / "tower_hazard_static_features.csv")[[
        "tower_id",
        "elevation_m",
        "slope_deg",
        "elevation_risk",
        "slope_risk",
    ]]
    frame = towers.merge(static, on="tower_id", how="left")
    for column in [
        "earthquake_score",
        "cyclone_score",
        "flood_history_score",
        "isolation_score",
        "elevation_risk",
        "slope_risk",
        "cell_count",
        POPULATION_COLUMN,
    ]:
        frame[column] = _numeric_optional(frame, column, 0.0)
    frame["earthquake_history_score"] = frame["earthquake_score"].clip(0, 1)
    frame["cyclone_history_score"] = frame["cyclone_score"].clip(0, 1)
    frame["flood_history_score"] = frame["flood_history_score"].clip(0, 1)
    frame["isolation_score"] = frame["isolation_score"].clip(0, 1)
    frame["elevation_risk"] = frame["elevation_risk"].clip(0, 1)
    frame["slope_risk"] = frame["slope_risk"].clip(0, 1)
    frame["radio_vulnerability"] = radio_vulnerability(frame.get("radios", pd.Series("", index=frame.index)))

    normalization = feature_normalization(frame)
    cell_count_max = float(normalization["cell_count_max"])
    cell_count = frame["cell_count"].clip(lower=1.0, upper=cell_count_max)
    frame["site_redundancy_risk"] = 1.0 - np.log1p(cell_count) / np.log1p(cell_count_max)
    frame["site_redundancy_risk"] = frame["site_redundancy_risk"].clip(0, 1)
    return frame, normalization


def expand_intensity(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for intensity in INTENSITIES:
        part = frame.copy()
        part["event_intensity"] = float(intensity)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def quake_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    expanded = expand_intensity(frame)
    history = expanded["earthquake_history_score"].to_numpy(float)
    event = expanded["event_intensity"].to_numpy(float)
    isolation = expanded["isolation_score"].to_numpy(float)
    radio = expanded["radio_vulnerability"].to_numpy(float)
    redundancy = expanded["site_redundancy_risk"].to_numpy(float)
    target = (
        0.43 * history
        + 0.31 * event
        + 0.10 * isolation
        + 0.08 * radio
        + 0.08 * redundancy
        + 0.13 * history * event
    )
    expanded["target"] = np.clip(target, 0, 1)
    features = [
        "earthquake_history_score",
        "event_intensity",
        "isolation_score",
        "radio_vulnerability",
        "site_redundancy_risk",
    ]
    return expanded, features


def cyclone_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    expanded = expand_intensity(frame)
    history = expanded["cyclone_history_score"].to_numpy(float)
    event = expanded["event_intensity"].to_numpy(float)
    elevation = expanded["elevation_risk"].to_numpy(float)
    flood = expanded["flood_history_score"].to_numpy(float)
    isolation = expanded["isolation_score"].to_numpy(float)
    radio = expanded["radio_vulnerability"].to_numpy(float)
    target = (
        0.34 * history
        + 0.28 * event
        + 0.12 * elevation
        + 0.08 * flood
        + 0.08 * isolation
        + 0.05 * radio
        + 0.14 * history * event
    )
    expanded["target"] = np.clip(target, 0, 1)
    features = [
        "cyclone_history_score",
        "event_intensity",
        "elevation_risk",
        "flood_history_score",
        "isolation_score",
        "radio_vulnerability",
    ]
    return expanded, features


def model_params() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": 220,
        "learning_rate": 0.055,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.86,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.03,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
        "eval_metric": "rmse",
    }


def _fit_regressor(data: pd.DataFrame, features: list[str], params: dict[str, Any] | None = None) -> XGBRegressor:
    model = XGBRegressor(**(params or model_params()))
    model.fit(data[features].astype(float), data["target"].to_numpy(float))
    return model


def cross_validate(data: pd.DataFrame, features: list[str], group_column: str = "adm3_name") -> dict[str, Any]:
    groups = data[group_column].astype(str)
    if groups.nunique() < 3:
        raise ValueError(f"At least three {group_column} groups are required for geographic CV")
    predictors = data[features].astype(float)
    target = data["target"].to_numpy(float)
    metrics = []
    for fold, (train, validation) in enumerate(GroupKFold(n_splits=3).split(predictors, target, groups), 1):
        model = _fit_regressor(data.iloc[train], features)
        prediction = np.clip(model.predict(predictors.iloc[validation]), 0, 1)
        validation_groups = sorted(groups.iloc[validation].unique().tolist())
        metrics.append({
            "fold": fold,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "validation_groups": validation_groups,
            "mae": float(mean_absolute_error(target[validation], prediction)),
            "rmse": float(mean_squared_error(target[validation], prediction) ** 0.5),
            "r2": float(r2_score(target[validation], prediction)),
        })
    return {
        "method": "GroupKFold by township",
        "group_column": group_column,
        "n_splits": 3,
        "folds": metrics,
        "mae_mean": float(np.mean([metric["mae"] for metric in metrics])),
        "rmse_mean": float(np.mean([metric["rmse"] for metric in metrics])),
        "r2_mean": float(np.mean([metric["r2"] for metric in metrics])),
        "metrics_interpretation": "proxy_formula_reproduction_only",
    }


def train_one(name: str, data: pd.DataFrame, features: list[str], path: Path) -> dict[str, Any]:
    cv = cross_validate(data, features)
    model = _fit_regressor(data, features)
    model.save_model(path)
    return {
        "name": name,
        "model_file": path.name,
        "model_sha256": _sha256(path),
        "features_in_order": features,
        "training_rows": len(data),
        "training_towers": int(data["tower_id"].nunique()),
        "synthetic_event_intensity_levels": INTENSITIES.tolist(),
        "cv_summary": cv,
        "hyperparameters": model_params(),
        "metrics_interpretation": "proxy_formula_reproduction_only; not disaster or outage forecasting skill",
    }


def compound_scenario_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for scenario_id, scenario in enumerate(COMPOUND_SCENARIOS):
        part = frame.copy()
        part["compound_scenario_id"] = scenario_id
        for name, value in scenario.items():
            part[name] = float(value)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def compound_dataset(
    frame: pd.DataFrame,
    *,
    flood_model: XGBRegressor,
    flood_features: list[str],
    quake_model: XGBRegressor,
    quake_features: list[str],
    cyclone_model: XGBRegressor,
    cyclone_features: list[str],
) -> pd.DataFrame:
    """Create compound rows using the exact runtime feature contracts."""
    expanded = compound_scenario_frame(frame)
    flood_frame = pd.DataFrame({
        "rain_30d_percentile": expanded["rain_30d_percentile"],
        "flood_history_score": expanded["flood_history_score"],
        "elevation_risk": expanded["elevation_risk"],
        "slope_risk": expanded["slope_risk"],
    })
    quake_frame = pd.DataFrame({
        "earthquake_history_score": expanded["earthquake_history_score"],
        "event_intensity": expanded["earthquake_event_intensity"],
        "isolation_score": expanded["isolation_score"],
        "radio_vulnerability": expanded["radio_vulnerability"],
        "site_redundancy_risk": expanded["site_redundancy_risk"],
    })
    cyclone_frame = pd.DataFrame({
        "cyclone_history_score": expanded["cyclone_history_score"],
        "event_intensity": expanded["cyclone_event_intensity"],
        "elevation_risk": expanded["elevation_risk"],
        "flood_history_score": expanded["flood_history_score"],
        "isolation_score": expanded["isolation_score"],
        "radio_vulnerability": expanded["radio_vulnerability"],
    })
    expanded["flood_ai_score"] = np.clip(flood_model.predict(flood_frame[flood_features]), 0, 1)
    expanded["earthquake_ai_score"] = np.clip(quake_model.predict(quake_frame[quake_features]), 0, 1)
    expanded["cyclone_ai_score"] = np.clip(cyclone_model.predict(cyclone_frame[cyclone_features]), 0, 1)

    flood = expanded["flood_ai_score"].to_numpy(float)
    quake = expanded["earthquake_ai_score"].to_numpy(float)
    cyclone = expanded["cyclone_ai_score"].to_numpy(float)
    isolation = expanded["isolation_score"].to_numpy(float)
    top = np.sort(np.column_stack([flood, quake, cyclone]), axis=1)
    union = 1.0 - (1.0 - 0.58 * flood) * (1.0 - 0.52 * quake) * (1.0 - 0.55 * cyclone)
    expanded["target"] = np.clip(union + 0.10 * top[:, 2] * top[:, 1] + 0.06 * isolation, 0, 1)
    return expanded


def _fit_upstream_models(
    training_base: pd.DataFrame,
    flood_training: pd.DataFrame,
    *,
    flood_features: list[str],
    flood_params: dict[str, Any],
    quake_features: list[str],
    cyclone_features: list[str],
) -> tuple[XGBRegressor, XGBRegressor, XGBRegressor]:
    training_tower_ids = set(training_base["tower_id"].astype(int))
    flood_rows = flood_training[flood_training["tower_id"].astype(int).isin(training_tower_ids)].copy()
    if flood_rows.empty:
        raise ValueError("Outer training fold contains no Flood model rows")
    flood_rows = flood_rows.rename(columns={"hazard_target_score": "target"})
    flood_model = _fit_regressor(flood_rows, flood_features, flood_params)
    quake_data, _ = quake_training(training_base)
    cyclone_data, _ = cyclone_training(training_base)
    quake_model = _fit_regressor(quake_data, quake_features)
    cyclone_model = _fit_regressor(cyclone_data, cyclone_features)
    return flood_model, quake_model, cyclone_model


def compound_cross_validation(
    frame: pd.DataFrame,
    flood_training: pd.DataFrame,
    *,
    flood_features: list[str],
    flood_params: dict[str, Any],
    quake_features: list[str],
    cyclone_features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Outer geographic CV with every upstream model isolated from held-out groups."""
    groups = frame["adm3_name"].astype(str)
    metrics = []
    cross_fitted_parts = []
    splitter = GroupKFold(n_splits=3)
    for fold, (train, validation) in enumerate(splitter.split(frame, groups=groups), 1):
        training_base = frame.iloc[train].copy()
        validation_base = frame.iloc[validation].copy()
        training_groups = set(training_base["adm3_name"].astype(str))
        validation_groups = set(validation_base["adm3_name"].astype(str))
        training_towers = set(training_base["tower_id"].astype(int))
        validation_towers = set(validation_base["tower_id"].astype(int))
        if training_groups.intersection(validation_groups) or training_towers.intersection(validation_towers):
            raise AssertionError("Outer compound fold leaked a held-out township or tower")

        flood_model, quake_model, cyclone_model = _fit_upstream_models(
            training_base,
            flood_training,
            flood_features=flood_features,
            flood_params=flood_params,
            quake_features=quake_features,
            cyclone_features=cyclone_features,
        )
        training_compound = compound_dataset(
            training_base,
            flood_model=flood_model,
            flood_features=flood_features,
            quake_model=quake_model,
            quake_features=quake_features,
            cyclone_model=cyclone_model,
            cyclone_features=cyclone_features,
        )
        validation_compound = compound_dataset(
            validation_base,
            flood_model=flood_model,
            flood_features=flood_features,
            quake_model=quake_model,
            quake_features=quake_features,
            cyclone_model=cyclone_model,
            cyclone_features=cyclone_features,
        )
        compound_model = _fit_regressor(training_compound, COMPOUND_FEATURES)
        prediction = np.clip(compound_model.predict(validation_compound[COMPOUND_FEATURES]), 0, 1)
        target = validation_compound["target"].to_numpy(float)
        metrics.append({
            "fold": fold,
            "train_towers": len(training_towers),
            "validation_towers": len(validation_towers),
            "train_groups": sorted(training_groups),
            "validation_groups": sorted(validation_groups),
            "upstream_validation_tower_overlap": 0,
            "upstream_validation_group_overlap": 0,
            "mae": float(mean_absolute_error(target, prediction)),
            "rmse": float(mean_squared_error(target, prediction) ** 0.5),
            "r2": float(r2_score(target, prediction)),
        })
        cross_fitted_parts.append(validation_compound)

    cross_fitted = pd.concat(cross_fitted_parts, ignore_index=True)
    if cross_fitted["tower_id"].nunique() != frame["tower_id"].nunique():
        raise AssertionError("Cross-fitted compound rows do not cover every training tower")
    return {
        "method": "outer GroupKFold by township with fold-local Flood, Earthquake, and Cyclone upstream fits",
        "group_column": "adm3_name",
        "n_splits": 3,
        "upstream_holdout_policy": "no validation township or tower used to fit any upstream model",
        "folds": metrics,
        "mae_mean": float(np.mean([metric["mae"] for metric in metrics])),
        "rmse_mean": float(np.mean([metric["rmse"] for metric in metrics])),
        "r2_mean": float(np.mean([metric["r2"] for metric in metrics])),
        "metrics_interpretation": "end-to-end proxy_formula_reproduction_only",
    }, cross_fitted


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    frame, normalization = base_frame()
    quake_data, quake_features = quake_training(frame)
    cyclone_data, cyclone_features = cyclone_training(frame)

    quake_path = MODELS / "hazard_earthquake_xgb.json"
    cyclone_path = MODELS / "hazard_cyclone_xgb.json"
    compound_path = MODELS / "hazard_compound_xgb.json"
    quake_metadata = train_one("GeoVision Earthquake Impact AI", quake_data, quake_features, quake_path)
    cyclone_metadata = train_one("GeoVision Cyclone Impact AI", cyclone_data, cyclone_features, cyclone_path)

    flood_metadata_path = MODELS / "hazard_model_metadata.json"
    flood_metadata = json.loads(flood_metadata_path.read_text(encoding="utf-8"))
    flood_features = list(flood_metadata["features_in_order"])
    flood_params = dict(flood_metadata["hyperparameters"])
    flood_training_path = DATA / "hazard_training_data.csv"
    flood_training = pd.read_csv(flood_training_path)

    compound_cv, cross_fitted = compound_cross_validation(
        frame,
        flood_training,
        flood_features=flood_features,
        flood_params=flood_params,
        quake_features=quake_features,
        cyclone_features=cyclone_features,
    )
    compound_model = _fit_regressor(cross_fitted, COMPOUND_FEATURES)
    compound_model.save_model(compound_path)
    compound_metadata = {
        "name": "GeoVision Compound Disaster Impact AI",
        "model_file": compound_path.name,
        "model_sha256": _sha256(compound_path),
        "features_in_order": COMPOUND_FEATURES,
        "training_rows": len(cross_fitted),
        "training_towers": int(cross_fitted["tower_id"].nunique()),
        "proxy_scenarios": list(COMPOUND_SCENARIOS),
        "training_feature_generation": {
            "flood": "geographically cross-fitted XGBRegressor using hazard_model_metadata.json feature contract",
            "earthquake": "geographically cross-fitted Earthquake Impact AI prediction",
            "cyclone": "geographically cross-fitted Cyclone Impact AI prediction",
        },
        "cv_summary": compound_cv,
        "hyperparameters": model_params(),
        "metrics_interpretation": "proxy_formula_reproduction_only; not joint probability or outage forecasting skill",
    }

    metadata = {
        "system_name": "GeoVision Disaster Impact AI — Multi-Hazard Model 2",
        "feature_normalization": normalization,
        "hazards": {
            "earthquake": quake_metadata,
            "cyclone": cyclone_metadata,
            "compound": compound_metadata,
        },
        "common_thresholds": {"high": 0.50, "very_high": 0.70},
        "training_inputs": {
            "tower_sites_population.csv": _sha256(DATA / "tower_sites_population.csv"),
            "tower_hazard_static_features.csv": _sha256(DATA / "tower_hazard_static_features.csv"),
            "hazard_training_data.csv": _sha256(flood_training_path),
            "hazard_model_metadata.json": _sha256(flood_metadata_path),
        },
        "versions": {
            "python": sys.version.split()[0],
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Earthquake, cyclone and compound targets are transparent pseudo-labels, not calibrated physical probabilities.",
            "Metrics measure reproduction of authored proxy logic; they do not measure disaster, tower-outage, or service-impact forecasting skill.",
            "Earthquake training uses the project's historical earthquake exposure index; operational assessment requires authoritative shaking products and verified tower outcomes.",
            "Cyclone history is limited to the uploaded local record; the packaged model has not been retrained on full IBTrACS or verified cyclone/outage labels.",
            "Compound validation isolates complete townships and fits every upstream model without held-out towers or geographies.",
            "Compound risk is a meta-model over three hazard-AI outputs plus site isolation; it is a planning exposure score, not a joint physical probability.",
        ],
    }
    (MODELS / "multi_hazard_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "feature_normalization": normalization,
        "earthquake_cv": quake_metadata["cv_summary"],
        "cyclone_cv": cyclone_metadata["cv_summary"],
        "compound_cv": compound_cv,
    }, indent=2))


if __name__ == "__main__":
    main()
