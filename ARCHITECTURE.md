# GeoVision AI Architecture

## Runtime flow

```text
Streamlit UI
    │
    ├── Model 1: site_ai compatibility adapter
    │       └── models/TowerRecommendationModel
    │               ├── validated feature contract
    │               ├── XGBoost probability
    │               └── native TreeSHAP reason
    │
    └── Model 2: engine/HazardEngine
            ├── data/preprocessing
            ├── hazard_ai compatibility runtime
            │       ├── cached GEE connector
            │       ├── cached USGS connector
            │       ├── cached keyless JTWC connector
            │       └── local packaged fallback
            ├── models/{Flood,Cyclone,Earthquake,Compound}Model
            └── engine/DecisionEngine
                    ├── risk category
                    ├── impact fields unknown until coverage simulation
                    ├── recommended action
                    └── native TreeSHAP factors

Coverage scenario (selected radius + score threshold)
    └── simulate_coverage: cached neighbors + complete surviving-tower search
            └── attach_coverage_scenario: affected = rerouted + losing coverage
```

## Data refresh policy

| Data | Source | Policy |
|---|---|---|
| Elevation/slope | GEE SRTM | Static/permanent cache |
| Land cover | GEE Dynamic World | Recent quality mosaic; confidence filtered and cached for 6 hours |
| Rainfall | GEE GSMaP | 15-minute cache; 48-hour maximum source age; complete 24/72/720-hour source and pixel observations |
| Current cyclone position/forecast | JTWC operational ATCF forecast products via U.S. Naval Research Laboratory | Short cache; 36-hour advisory and 48-hour index age policies; no API key |
| Historical cyclone research | NOAA IBTrACS | Intended archive for the next historical feature/retraining pipeline; not presented as a live source |
| Earthquakes | USGS FDSN | 5-minute cache |
| Tower/population/flood context | Packaged project data | Loaded once per app process |

## Model contracts

The v8 hazard result view lives in `ui/hazard_results.py`. `utils/result_context.py` derives safe, explicit evidence metadata from actual runtime modes; it does not infer confidence from model scores or provider names. `utils/scenario_report.py` validates per-tower/aggregate population accounting and attaches the selected assumptions to CSV/JSON outputs. Raw connector errors are excluded from structured evidence notes.

The serving wrappers read feature order from model metadata and reject missing, null, or non-numeric features. Scores are clipped to `[0, 1]`. Explanations use XGBoost's `pred_contribs`, which computes exact TreeSHAP contributions in raw-margin space without a separate SHAP runtime dependency.

Compound child outputs must contain exactly one valid row per requested tower, with finite `[0, 1]` scores. Alignment is by tower ID, never row position; missing child scores are rejected rather than replaced by zero. `DecisionEngine` also rejects invalid scores and derives exposure categories consistently from their numeric thresholds. Guidance is conditional engineering review, not an automated operational dispatch.

`multi_hazard_model_metadata.json` stores fixed normalizers from the complete packaged training universe. Runtime normalization never divides by the currently selected batch maximum. Earthquake, cyclone and compound artifacts include hashes and training-input provenance. Compound outer validation holds out complete townships from every upstream fit; final compound training rows use geographically cross-fitted upstream predictions.

`DecisionEngine.enrich` preserves the impact-column names but returns unknown (`NaN`) counts with `impact_status="requires_coverage_scenario"`. Call `attach_coverage_scenario(predictions, tower_load)` after `simulate_coverage` to obtain actual scenario counts and `impact_status="coverage_scenario"`. Counts are attributed to each population cell's original primary tower; changing the radius changes that attribution. The dashboard does this on every scenario-control rerun and hides results when hazard, area or source mode changes.

## Scientific boundaries

- Tower recommendation labels reproduce a hand-written planning score.
- Flood labels reproduce a formula based on rainfall percentile, flood history, elevation, and slope.
- Earthquake and cyclone labels combine historical indices with synthetic event intensities.
- Compound labels are synthetic functions of the three submodel outputs.
- Population impact is nearest-tower geographic screening, not RF propagation, capacity, or observed handover.
- The selected-radius scenario uses all surviving towers, with unassessed towers assumed available. The fixed 5 km population model feature is not the selected-radius impact count.
- Dynamic World is displayed as context, not a learned Flood feature. The Cyclone model does not yet consume live GEE rainfall or Dynamic World.
- Local cyclone history is background only; a decades-old archive point is never presented as an active storm.
- JTWC data access is a public operational-product adapter, not a guaranteed JSON API or service-level agreement. Auto mode has a local fallback.
- The packaged Cyclone AI has not yet been retrained on a full IBTrACS history; its historical exposure proxy remains limited.
- USGS reports earthquakes; the system estimates post-event impact and never claims earthquake prediction.

Before operational use, replace pseudo-labels with matched disaster observations and tower outage/restoration logs, hold out complete events and geographic regions, calibrate scores, test drift, and establish human approval procedures.

## Incremental migration

The original `site_ai.py` and `hazard_ai.py` APIs are intentionally retained. New UI code calls `HazardEngine`; the engine currently adapts the proven legacy live/local runtime and adds standardized decisions and explanations. This minimizes regression risk while allowing connectors and submodels to move behind their new interfaces independently.
