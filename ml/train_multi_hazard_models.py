"""Train GeoVision Disaster Impact AI submodels for earthquake, cyclone and compound risk.

These are MVP/pseudo-label models. They use the project's existing historical exposure
indices plus transparent tower-vulnerability features and synthetic event-intensity
levels so the runtime can accept live/current event signals. They are NOT calibrated
physical disaster probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = BASE / "models"

RANDOM_STATE = 42
INTENSITIES = np.array([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], dtype=float)


def radio_vulnerability(s: pd.Series) -> np.ndarray:
    text = s.fillna("").astype(str).str.upper()
    tech_count = text.str.count(",") + 1
    has_lte = text.str.contains("LTE").astype(float)
    has_umts = text.str.contains("UMTS").astype(float)
    # Single-technology/legacy sites get a slightly higher vulnerability proxy;
    # multi-RAT sites are treated as more resilient. This is an MVP proxy only.
    value = 0.74 - 0.12 * (tech_count - 1) - 0.10 * has_lte - 0.03 * has_umts
    return np.clip(value.to_numpy(float), 0.25, 0.80)


def base_frame() -> pd.DataFrame:
    towers = pd.read_csv(DATA / "tower_sites_population.csv")
    static = pd.read_csv(DATA / "tower_hazard_static_features.csv")[[
        "tower_id", "elevation_m", "slope_deg", "elevation_risk", "slope_risk"
    ]]
    df = towers.merge(static, on="tower_id", how="left")
    for c in [
        "earthquake_score", "cyclone_score", "flood_history_score", "isolation_score",
        "elevation_risk", "slope_risk", "cell_count"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["earthquake_history_score"] = df["earthquake_score"].clip(0, 1)
    df["cyclone_history_score"] = df["cyclone_score"].clip(0, 1)
    df["flood_history_score"] = df["flood_history_score"].clip(0, 1)
    df["isolation_score"] = df["isolation_score"].clip(0, 1)
    df["elevation_risk"] = df["elevation_risk"].clip(0, 1)
    df["slope_risk"] = df["slope_risk"].clip(0, 1)
    df["radio_vulnerability"] = radio_vulnerability(df["radios"])
    max_cells = max(float(df["cell_count"].max()), 1.0)
    df["site_redundancy_risk"] = 1.0 - np.log1p(df["cell_count"].clip(lower=1)) / np.log1p(max_cells)
    df["site_redundancy_risk"] = df["site_redundancy_risk"].clip(0, 1)
    return df


def expand_intensity(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for intensity in INTENSITIES:
        p = df.copy()
        p["event_intensity"] = float(intensity)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def quake_training(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = expand_intensity(df)
    h = x["earthquake_history_score"].to_numpy(float)
    e = x["event_intensity"].to_numpy(float)
    iso = x["isolation_score"].to_numpy(float)
    radio = x["radio_vulnerability"].to_numpy(float)
    redund = x["site_redundancy_risk"].to_numpy(float)
    # Pseudo impact target: historical seismic exposure + current event signal + tower fragility.
    y = 0.43*h + 0.31*e + 0.10*iso + 0.08*radio + 0.08*redund + 0.13*h*e
    x["target"] = np.clip(y, 0, 1)
    features = ["earthquake_history_score", "event_intensity", "isolation_score", "radio_vulnerability", "site_redundancy_risk"]
    return x, features


def cyclone_training(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = expand_intensity(df)
    h = x["cyclone_history_score"].to_numpy(float)
    e = x["event_intensity"].to_numpy(float)
    elev = x["elevation_risk"].to_numpy(float)
    flood = x["flood_history_score"].to_numpy(float)
    iso = x["isolation_score"].to_numpy(float)
    radio = x["radio_vulnerability"].to_numpy(float)
    # Pseudo impact target: historical cyclone exposure + current track/wind signal,
    # amplified in low terrain/flood-prone and isolated sites.
    y = 0.34*h + 0.28*e + 0.12*elev + 0.08*flood + 0.08*iso + 0.05*radio + 0.14*h*e
    x["target"] = np.clip(y, 0, 1)
    features = ["cyclone_history_score", "event_intensity", "elevation_risk", "flood_history_score", "isolation_score", "radio_vulnerability"]
    return x, features


def model_params() -> dict:
    return dict(
        objective="reg:squarederror",
        n_estimators=220,
        learning_rate=0.055,
        max_depth=4,
        min_child_weight=3,
        subsample=0.86,
        colsample_bytree=0.90,
        reg_alpha=0.03,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
        eval_metric="rmse",
    )


def cross_validate(data: pd.DataFrame, features: list[str]) -> dict:
    groups = data["tower_id"].to_numpy()
    X = data[features].astype(float)
    y = data["target"].to_numpy(float)
    metrics = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=3).split(X, y, groups), 1):
        m = XGBRegressor(**model_params())
        m.fit(X.iloc[tr], y[tr])
        pred = np.clip(m.predict(X.iloc[va]), 0, 1)
        metrics.append({
            "fold": fold,
            "mae": float(mean_absolute_error(y[va], pred)),
            "rmse": float(mean_squared_error(y[va], pred) ** 0.5),
            "r2": float(r2_score(y[va], pred)),
        })
    return {
        "folds": metrics,
        "mae_mean": float(np.mean([m["mae"] for m in metrics])),
        "rmse_mean": float(np.mean([m["rmse"] for m in metrics])),
        "r2_mean": float(np.mean([m["r2"] for m in metrics])),
    }


def train_one(name: str, data: pd.DataFrame, features: list[str], path: Path) -> dict:
    cv = cross_validate(data, features)
    model = XGBRegressor(**model_params())
    model.fit(data[features].astype(float), data["target"].to_numpy(float))
    model.save_model(path)
    return {
        "name": name,
        "model_file": path.name,
        "features_in_order": features,
        "training_rows": int(len(data)),
        "training_towers": int(data["tower_id"].nunique()),
        "synthetic_event_intensity_levels": INTENSITIES.tolist(),
        "cv_summary": cv,
        "hyperparameters": model_params(),
    }


def compound_training(df: pd.DataFrame, quake_model: XGBRegressor, quake_features: list[str], cyclone_model: XGBRegressor, cyclone_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    # Create paired event conditions, including single-hazard and multi-hazard cases.
    rng = np.random.default_rng(RANDOM_STATE)
    reps = 6
    parts = []
    for r in range(reps):
        p = df.copy()
        if r == 0:
            p["flood_ai_score"] = np.clip(0.35*p["flood_history_score"] + 0.25*p["elevation_risk"], 0, 1)
            p["earthquake_event_intensity"] = 0.0
            p["cyclone_event_intensity"] = 0.0
        else:
            p["flood_ai_score"] = np.clip(
                0.20*p["flood_history_score"] + 0.15*p["elevation_risk"] + rng.uniform(0.0, 1.0, len(p))*0.75,
                0, 1,
            )
            p["earthquake_event_intensity"] = rng.uniform(0.0, 1.0, len(p))
            p["cyclone_event_intensity"] = rng.uniform(0.0, 1.0, len(p))

        qx = pd.DataFrame({
            "earthquake_history_score": p["earthquake_history_score"],
            "event_intensity": p["earthquake_event_intensity"],
            "isolation_score": p["isolation_score"],
            "radio_vulnerability": p["radio_vulnerability"],
            "site_redundancy_risk": p["site_redundancy_risk"],
        })
        cx = pd.DataFrame({
            "cyclone_history_score": p["cyclone_history_score"],
            "event_intensity": p["cyclone_event_intensity"],
            "elevation_risk": p["elevation_risk"],
            "flood_history_score": p["flood_history_score"],
            "isolation_score": p["isolation_score"],
            "radio_vulnerability": p["radio_vulnerability"],
        })
        p["earthquake_ai_score"] = np.clip(quake_model.predict(qx[quake_features]), 0, 1)
        p["cyclone_ai_score"] = np.clip(cyclone_model.predict(cx[cyclone_features]), 0, 1)
        parts.append(p)
    x = pd.concat(parts, ignore_index=True)

    f = x["flood_ai_score"].to_numpy(float)
    q = x["earthquake_ai_score"].to_numpy(float)
    c = x["cyclone_ai_score"].to_numpy(float)
    iso = x["isolation_score"].to_numpy(float)
    top = np.sort(np.column_stack([f, q, c]), axis=1)
    # Nonlinear union + interaction: high values in multiple hazards raise compound impact.
    union = 1.0 - (1.0 - 0.58*f)*(1.0 - 0.52*q)*(1.0 - 0.55*c)
    y = union + 0.10*(top[:, 2]*top[:, 1]) + 0.06*iso
    x["target"] = np.clip(y, 0, 1)
    features = ["flood_ai_score", "earthquake_ai_score", "cyclone_ai_score", "isolation_score"]
    return x, features


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    df = base_frame()

    qdata, qfeatures = quake_training(df)
    cdata, cfeatures = cyclone_training(df)

    qmeta = train_one("GeoVision Earthquake Impact AI", qdata, qfeatures, MODELS / "hazard_earthquake_xgb.json")
    cmeta = train_one("GeoVision Cyclone Impact AI", cdata, cfeatures, MODELS / "hazard_cyclone_xgb.json")

    qmodel = XGBRegressor(); qmodel.load_model(MODELS / "hazard_earthquake_xgb.json")
    cmodel = XGBRegressor(); cmodel.load_model(MODELS / "hazard_cyclone_xgb.json")
    comp, comp_features = compound_training(df, qmodel, qfeatures, cmodel, cfeatures)
    compmeta = train_one("GeoVision Compound Disaster Impact AI", comp, comp_features, MODELS / "hazard_compound_xgb.json")

    metadata = {
        "system_name": "GeoVision Disaster Impact AI — Multi-Hazard Model 2",
        "hazards": {
            "earthquake": qmeta,
            "cyclone": cmeta,
            "compound": compmeta,
        },
        "common_thresholds": {"high": 0.50, "very_high": 0.70},
        "limitations": [
            "Earthquake, cyclone and compound targets are transparent pseudo-labels, not calibrated physical probabilities.",
            "Earthquake training uses the project's historical earthquake exposure index; operational shaking prediction requires ground-motion labels and/or authoritative shaking products.",
            "Cyclone training uses the project's historical cyclone exposure index, which is limited by the currently uploaded cyclone record. Live mode can add GEE IBTrACS track/wind context, but the GEE catalog is a best-track archive rather than a forecast model.",
            "Compound risk is a learned meta-model over the three hazard-AI outputs plus site isolation; it represents planning exposure, not a joint physical probability.",
            "Cross-validation measures reproduction of the MVP pseudo-label logic, not real-world disaster forecasting skill.",
        ],
    }
    (MODELS / "multi_hazard_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
