#!/usr/bin/env python3
"""Train Model 2: GeoVision Hazard AI flood/heavy-rain exposure prototype."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FEATURES = ["rain_30d_percentile", "flood_history_score", "elevation_risk", "slope_risk"]
TARGET = "hazard_target_score"

df = pd.read_csv(DATA / "hazard_training_data.csv")
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
groups = df["adm3_name"].astype(str)
params = dict(
    objective="reg:squarederror", n_estimators=260, learning_rate=0.05, max_depth=4,
    min_child_weight=3, subsample=0.85, colsample_bytree=0.9, reg_alpha=0.03,
    reg_lambda=2.0, tree_method="hist", random_state=42, n_jobs=2, eval_metric="rmse",
)

cv = GroupKFold(n_splits=5)
oof = np.zeros(len(df), dtype=float)
rows = []
for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
    model = XGBRegressor(**params)
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = np.clip(model.predict(X.iloc[te]), 0, 1)
    oof[te] = pred
    rows.append({
        "fold": fold,
        "train_rows": len(tr),
        "test_rows": len(te),
        "mae": mean_absolute_error(y.iloc[te], pred),
        "rmse": mean_squared_error(y.iloc[te], pred) ** 0.5,
        "r2": r2_score(y.iloc[te], pred),
    })

model = XGBRegressor(**params)
model.fit(X, y)
MODELS.mkdir(exist_ok=True)
model.save_model(MODELS / "hazard_flood_xgb.json")
pd.DataFrame(rows).to_csv(DATA / "hazard_xgb_cv_metrics.csv", index=False)

cvdf = pd.DataFrame(rows)
metadata = {
    "model_name": "GeoVision Hazard AI — Flood/Heavy-Rain Exposure Prototype",
    "model_type": "XGBRegressor",
    "target": TARGET,
    "features_in_order": FEATURES,
    "training_rows": int(len(df)),
    "training_towers": int(df.tower_id.nunique()),
    "training_snapshots": int(pd.to_datetime(df.date).nunique()),
    "high_exposure_threshold": 0.50,
    "very_high_exposure_threshold": 0.70,
    "runtime_gee_features": ["GSMaP 30-day rainfall", "SRTM elevation", "SRTM-derived slope"],
    "local_static_feature": "historic flood susceptibility from project flood footprint",
    "cv_summary": {
        "mae_mean": float(cvdf.mae.mean()),
        "rmse_mean": float(cvdf.rmse.mean()),
        "r2_mean": float(cvdf.r2.mean()),
    },
    "hyperparameters": params,
    "limitations": [
        "The current target is a pseudo-label, not a calibrated flood probability.",
        "CV measures reproduction of prototype hazard logic rather than real flood forecasting skill.",
        "Operational training requires date-and-location matched observed flood labels.",
        "This training script covers the Flood/Heavy-Rain submodel; the multi-hazard system also includes separate Earthquake, Cyclone and Compound models.",
    ],
}
(MODELS / "hazard_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata["cv_summary"], indent=2))
