# GeoVision Disaster Impact AI — Multi-Hazard Model Card

## Model 2 modules

| Hazard | Model file | Runtime hazard signal | Main static context |
|---|---|---|---|
| Flood / Heavy Rain | `hazard_flood_xgb.json` | GEE GSMaP recent rainfall | SRTM terrain + historic flood susceptibility |
| Earthquake | `hazard_earthquake_xgb.json` | USGS recent event magnitude/depth/distance signal | Historic seismic exposure + site isolation/redundancy |
| Cyclone | `hazard_cyclone_xgb.json` | GEE NOAA IBTrACS track/wind context | Historic cyclone exposure + low terrain/flood/isolation |
| Compound | `hazard_compound_xgb.json` | Outputs of the three models above | Site isolation |

## What the score means

All outputs are **0–1 planning exposure/impact scores**. They are not calibrated probabilities that a disaster will happen or that a tower will fail.

## Why earthquake is not GEE-only

Google Earth Engine is used for raster/geospatial environmental inputs and the IBTrACS cyclone archive. Recent earthquake-event information is queried from the USGS FDSN earthquake service because an earthquake catalog/event API is the appropriate source for current seismic events.

## Cyclone caveat

GEE `NOAA/IBTrACS/v4` is a best-track/archive observational dataset. It is useful for historical/observational cyclone context but is **not a forecast feed**. The project's local cyclone label file is also extremely limited, so cyclone scores must remain MVP/prototype claims.

## Compound model

Compound AI is a trained XGBoost meta-model over:

- Flood AI score
- Earthquake AI score
- Cyclone AI score
- Tower isolation score

It models nonlinear multi-hazard planning exposure. It is not a joint physical probability model.

## Required upgrade for operational use

Replace pseudo-label targets with verified date/location matched disaster observations and real tower outage/service logs. Validate by holding out complete disaster events and geographic areas.
