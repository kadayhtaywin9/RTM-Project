# GeoVision Hazard AI — Model Card

**File:** `hazard_flood_xgb.json`  
**Type:** XGBoost regressor  
**Task:** Flood/heavy-rain exposure scoring for existing telecom tower sites.

## Features

1. `rain_30d_percentile`
2. `flood_history_score`
3. `elevation_risk`
4. `slope_risk`

## Training set

- 128,464 rows
- 8,029 current tower sites
- 16 representative historical rainfall snapshots
- rainfall history spans 2022-01-01 to 2026-07-21 in the included project data

## Runtime GEE sources

- JAXA GSMaP v6 operational rainfall
- USGS SRTM elevation
- Earth Engine terrain slope derived from SRTM

## Output

`hazard_ai_score` in the range 0–1. This is a **prototype exposure score**, not a calibrated probability that a flood will happen or a tower will fail.

## Validation interpretation

Cross-validation metrics are extremely high because the target itself is a deterministic pseudo-label created from the same hazard factors. The metrics therefore test whether XGBoost can reproduce the MVP hazard scoring logic, not whether it predicts unseen real flood events.

## Required upgrade for operational claims

Use real date-and-location matched flood event labels (or verified satellite-derived flood masks), train on historical GEE feature snapshots, hold out entire events/areas, and report event-level/spatial validation.
