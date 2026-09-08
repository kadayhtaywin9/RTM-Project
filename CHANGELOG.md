# Changelog

## v8 — Clear Results & UI Update

- Reorganized hazard results around the exposure map, a three-site engineering-review shortlist, and a compact review queue. Detailed model statistics, distributions and source notes use expandable sections.
- Added visible live/mixed/demo/fallback/unknown evidence labels. Removed probability-style formatting from the hazard map and renamed assumed scenario failures to avoid suggesting confirmed outages.
- Added full assessed-tower CSV exports and JSON scenario notes, including source status/times, score scales, scenario assumptions, population baseline and interpretation limits.
- Rejected missing, duplicate or invalid compound inputs and invalid decision scores. Score categories derive from the validated score.
- Made recommendations conditional planning guidance requiring confirmation of official alerts and site conditions before intervention.
- Rebuilt the Windows launcher with directory-safe paths, explicit Python selection, complete dependency checks and visible setup/startup errors.
- Deduplicated timestamp parsing to avoid repeatedly parsing the same source time for thousands of towers.
- Retained existing model artifacts: these changes improve integrity and interpretation, not demonstrated real-world forecasting accuracy.

## v7 — Reliability Update

- Corrected coverage-loss calculations to consider every surviving tower beyond the ten-neighbor cache.
- Replaced score-weighted population estimates with counts from the selected coverage scenario, with affected/rerouted/lost conservation checks.
- Added fixed training-set normalization so selecting a different tower batch cannot change identical features.
- Retrained earthquake/cyclone/compound proxy models, with township-isolated upstream validation and cross-fitted compound training inputs.
- Added strict GSMaP freshness, sample completeness, hourly inventory and per-pixel observation checks.
- Added USGS response validation and earthquake-only querying; malformed responses fail instead of appearing as zero events.
- Added JTWC index freshness and Southern Hemisphere season-year handling; stale-only products cannot establish no active storm.
- Changed local cyclone output to historical background only, with current storm status unknown.
- Hid previous results when source/model/area settings change and displayed completion time.
- Preserved the clean light interface, OpenStreetMap maps, and university/team network header.

## University identity header

- Moved the university logo, university name, team name, and network artwork from the page footer to the top of the dashboard.
- Kept the identity strip responsive and displayed only once above the GeoVision hero.

## Live cyclone source repair and network footer

- Replaced the retired NOAA SSD hostname with NRL's keyless ATCF current-storm index and `.fst` forecast products.
- Added NRL warning-link discovery, explicit rejection-page validation, and coverage for valid no-storm responses.
- Preserved bounded retrieval and stale-product rejection, while making incomplete storm sets fail closed in Live-only mode and use the cached fallback in Automatic mode.
- Moved the university and team identity into a responsive dark footer with a lightweight inline network graphic.
- Removed duplicate university branding from the main dashboard hero.

## Clean interface refinement

- Restored the familiar OpenStreetMap basemap for the main maps and site checker.
- Replaced the decorative dark console treatment with a light, restrained planning-dashboard design.
- Simplified navigation, scenario controls, data-source labels, cards, typography, and spacing.
- Replaced raw live-source exceptions with concise guidance and collapsible technical details.

## Live cyclone and visual-console upgrade

- Replaced the misleading live IBTrACS path with keyless JTWC operational current/forecast ingestion and stale-product validation.
- Replaced static ESA WorldCover sampling with a refreshed, confidence-filtered Dynamic World composite.
- Added a compact hero, responsive KPI cards, and clearer navigation; the later clean-interface refinement replaced the dark visual treatment.
- Preserved deterministic local fallbacks and the existing trained-model feature contracts.
- Added offline parser, source-fallback, and connector tests.
- Kept the Cyclone AI limitation explicit: the packaged model still needs a full IBTrACS historical feature pipeline and verified outage labels before operational retraining.

## Production-style architecture upgrade

- Added connector, model-serving, orchestration, decision, cache, and visualization packages.
- Added cached GEE static/rainfall sampling and cached USGS/archive access.
- Added native XGBoost TreeSHAP explanations to tower and hazard outputs.
- Added stable tower recommendation and hazard decision contracts.
- Added Flood, Cyclone, Earthquake, and Compound Risk dashboard tabs.
- Added post-event earthquake fields and cyclone context fields.
- Added population impact, coverage impact, recommended action, and risk summaries.
- Added bounded runtime dependencies, developer dependencies, tests, and smoke tests.
- Added architecture, GEE, local setup, and Streamlit Cloud deployment documentation.
- Retained legacy compatibility entry points and all packaged data/model artifacts.
