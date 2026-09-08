# GeoVision Disaster Impact AI — Multi-Hazard Model Card

## Model 2 modules

| Hazard | Model file | Runtime hazard signal | Main static context |
|---|---|---|---|
| Flood / Heavy Rain | `hazard_flood_xgb.json` | GEE GSMaP recent rainfall | SRTM terrain + historic flood susceptibility; Dynamic World is displayed context only |
| Earthquake | `hazard_earthquake_xgb.json` | USGS recent event magnitude/depth/distance signal | Historic seismic exposure + site isolation/redundancy |
| Cyclone | `hazard_cyclone_xgb.json` | JTWC current position and forecast track/wind signal | Historic cyclone exposure + low terrain/flood/isolation |
| Compound | `hazard_compound_xgb.json` | Outputs of the three models above | Site isolation |

## What the score means

All outputs are **0–1 planning exposure/impact scores**. They are not calibrated probabilities that a disaster will happen or that a tower will fail.

## Why earthquake is not GEE-only

Google Earth Engine is used for raster/geospatial environmental inputs. Recent earthquake-event information is queried from the USGS FDSN earthquake service because an earthquake catalog/event API is the appropriate source for current seismic events. Current cyclone products come from JTWC through the public U.S. Naval Research Laboratory ATCF feed and require no API key.

## Cyclone caveat

The JTWC adapter supplies current/forecast event context; it does not change the trained model or guarantee availability of the upstream public product site. The project's historical cyclone exposure input is still derived from an extremely limited local record. IBTrACS is the recommended archive for a future historical feature pipeline, but the packaged model has **not** yet been retrained on a full IBTrACS history. Cyclone scores must therefore remain MVP/prototype planning claims.

The v7 local/demo cyclone path uses historical background with a zero current-event signal and explicitly unknown current storm status. Live GEE rainfall and Dynamic World are not cyclone-model inputs in this release.

## Compound model

Compound AI is a trained XGBoost meta-model over:

- Flood AI score
- Earthquake AI score
- Cyclone AI score
- Tower isolation score

It models nonlinear multi-hazard planning exposure. It is not a joint physical probability model.

## v7 training and evaluation

Earthquake, cyclone and compound artifacts were retrained using the existing transparent proxy targets. Each has 48,174 rows representing 8,029 towers and six synthetic scenarios. Earthquake/cyclone validation uses three township-grouped folds. Compound validation refits Flood, Earthquake and Cyclone upstream models on each outer training fold only; no held-out township or tower is used in those fits. Final compound training uses the held-out upstream predictions assembled across folds, with the real Flood feature contract rather than a random rainfall heuristic.

Scores and reported MAE/RMSE/R² measure reproduction of authored proxy logic. They are not accuracy estimates against observed storms, outages or population service loss. Exact metrics, fold membership, hashes, dependency versions and fixed normalization parameters are recorded in `multi_hazard_model_metadata.json`.

The outer validation compound estimator trains on in-sample upstream outputs within its training fold, whereas the final compound estimator trains on cross-fitted upstream outputs. Outer held-out groups remain isolated, but this is not a fully nested evaluation of the final training procedure.

## Population interpretation

Population counts are calculated separately from model scores using the selected radius and tower-unavailability threshold. Initially affected population equals rerouted population plus population losing geographic coverage. The ten cached neighbors accelerate lookup but do not limit the set of eligible surviving towers. Unassessed towers are assumed available; RF, capacity, backhaul and power constraints are not simulated.

## Required upgrade for operational use

Replace pseudo-label targets with verified date/location matched disaster observations and real tower outage/service logs. Validate by holding out complete disaster events and geographic areas.
