# GeoVision AI v7 — Reliability Update

## Included

- Clean light dashboard, familiar OpenStreetMap maps, and the university logo/name plus Team GeoVisionaries in the network-style header.
- Corrected coverage simulation: if all ten cached neighbors fail, search all surviving towers within the selected radius before declaring coverage lost.
- Consistent scenario population counts in tables, charts and summary metrics. Initially affected = rerouted + losing coverage. No multiplication of an uncalibrated exposure score by population.
- Batch-independent normalization, refreshed earthquake/cyclone/compound artifacts, township-isolated upstream validation, and cross-fitted compound training inputs.
- Stronger JTWC, USGS and GSMaP validation, with explicit errors in Live only and labelled fallback in Automatic.
- Historical-only cyclone demo context; stale archive events cannot appear as active storms.
- Result timestamps and protection against displaying a previous demo result after switching to Live only.

## Start

Extract the ZIP into a new folder; keep an older copy for rollback. Follow `README.md` to install dependencies, then run:

```bash
streamlit run app.py
```

Demo data works without external credentials. Live flood sampling requires your Earth Engine project/authentication. JTWC and USGS connectors do not require API keys, but public-feed availability is outside this application's control.

## Compatibility

Deploy the complete folder, including updated model JSON files and metadata. Python modules implement training/inference; trained XGBoost artifacts are the JSON files in `models/`.

For code integrations, `DecisionEngine.enrich` still includes `population_affected` and `coverage_impact`, but returns `NaN` until a coverage scenario is attached. Use `engine.decision_engine.attach_coverage_scenario(predictions, tower_load)` with the `tower_load` returned by `geoai_engine.simulate_coverage`. Do not convert unknown impacts to zero.

## Important limitations

These remain research/demo planning models trained on proxy labels, not calibrated disaster or tower-failure probabilities. Improved validation does not establish observed forecasting accuracy. Full IBTrACS historical training and verified tower-outage labels are not included. The cyclone model does not yet use live GEE rainfall or Dynamic World; Dynamic World is contextual information in the Flood workspace, not a learned Flood feature.

Coverage is geographic nearest-tower screening using the packaged 2020 population baseline. It does not model RF propagation, capacity, power or backhaul failures. Unassessed towers outside selected analysis areas are assumed available.

## Verification

Release checks on 2026-09-06:

- 92 regression tests passed (nine pandas/NumPy timedelta deprecation warnings).
- All model smoke checks and all four Streamlit hazard workspaces passed, including radius changes and source-switch result invalidation.
- Scoped Ruff checks passed; `pip check` found no broken requirements.
- Model and training-input hashes matched metadata; full/subset/single-tower feature and earthquake-score invariance checks passed.
- Public JTWC and USGS connectors returned valid live responses at approximately 04:12 UTC. Feed availability can change after this check.
- Authenticated live GEE sampling was not run; GEE validation and API composition were exercised with offline fixtures.

Run the included regression and smoke checks:

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/smoke_test.py
python scripts/app_smoke_test.py
```

Regression tests exercise malformed/stale sources offline; authenticated live Earth Engine sampling must be checked in your configured deployment. Rainfall completeness uses Earth Engine's documented [per-pixel valid observation count](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-count) and [masking](https://developers.google.com/earth-engine/apidocs/ee-image-updatemask) APIs.
