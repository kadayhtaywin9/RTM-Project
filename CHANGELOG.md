# Changelog

## v21 — Live Assessment Gates & Restored Population Impact

- Fail-closed live-assessment rules cover Flood, Earthquake, Cyclone and Compound Risk. Unknown input is never treated as zero risk; no-event source checks do not display historical-only event scores.
- Skip earthquake/cyclone inference on no-event responses and skip the compound meta-model when any required component has no usable assessment.
- Added a gray availability map with null scores/impact exports and cleared prior results after source failures. No-event source labels are distinct from usable live-event inputs.
- Restored the hypothetical coverage-impact panel by default for eligible results, including a prominent initially-affected population total and the rerouted/lost-coverage split. Widened population cards after browser QA found truncation.
- Preserved all saved models, input datasets, 73-hour rainfall policy, completeness checks, SOS receipt/clock features and the no-bandwidth-mask Overview.
- 320 Python tests passed (264 retained plus 56 new); all four hazard app checks passed for demo/live fixtures, no-event/unavailable inputs, population totals and stale-result clearing. Clock tests and focused lint passed. Real provider availability and model accuracy were not verified.

## v20 — Source Transparency & SOS Receipt Times

- Removed bandwidth mask UI, capacity-scenario/interpolation modules and their feature-specific checks. Historical source data and past release notes are retained; v19 remains the recovery copy.
- Added server receipt date/time (Myanmar time, seconds) to SOS incident rows and selected-incident details; missing/ambiguous times stay unavailable.
- Added source counts, observation/query periods and retrieval clocks for GEE, USGS and JTWC, carried into analysis JSON exports. Overlapping rainfall windows are not double-counted; tower sampling is not represented as independent sensor observations.
- Raised GSMaP latest-image age tolerance from 48 to 73 hours. Data over 48 hours is explicitly delayed; accumulation periods and completeness validation are unchanged.
- Replaced stepped saturated hazard colors with a muted continuous fixed scale. Scores, thresholds and model artifacts are unchanged.
- Made hypothetical outage/population consequences optional, off by default, rather than automatically treating high scores as current failures.
- Added earthquake/cyclone zero-event reference scores and signed model-input changes. Removed the invented earthquake observation timestamp when no event time exists.
- Added reproducible offline model sensitivity results and explicit limits on accuracy/bias claims. 264 Python tests and a four-hazard full-app smoke check passed at release verification.

## v19 — Classic Towers & Live SOS Clock

- Restored compact, semi-transparent blue tower points, detailed hover metadata, study-area boundaries, gap markers and ranked suggested sites in Overview.
- Corrected Overview to plot the selected area's towers instead of the full inventory. The fixed 2,500-point display sample is disclosed; calculations retain the selected inventory.
- Restored the separate bandwidth evidence button and the existing collected-measurement, full-AOI estimate, planning-scenario and manual-upload views. The v18 distance-only Mbps overlay is no longer the Overview entry point.
- Added a prominent upper-right resident SOS clock: explicit Myanmar time, seconds, local date, timezone label, immediate display and refresh on return to the page. It is a device clock, not a server-synchronized emergency timestamp.
- Moved the five-second Streamlit fragment timer onto the incident feed, where Locate/Acknowledge/Resolve actions run.
- Added static clock serving, no-store resident HTML, targeted API/map/timer tests, a deterministic clock test and a Streamlit UI smoke test. Ignored local SOS databases in Git.
- Preserved datasets, model artifacts and existing SOS endpoints. Documented unresolved deployment and model limitations in PROJECT_REVIEW_V19.md.

## v14 — Full AOI Estimates

- Added an explicit estimated map filling all selected project polygons with a 500 m or 1000 m grid, independent of population-pixel availability.
- Added a button from collected measurements that carries the selected quarter and minimum-test filter into the new view.
- Uses inverse-distance-squared weighting of up to eight nearby source tile averages. Original observations, counts and model artifacts remain unchanged.
- Added source-distance coloring, interpolation/extrapolation flags, area-weighted summaries, a spatial holdout comparison and separately labelled CSV/GeoJSON exports with source attribution.
- Default Q2/minimum-three validation has 20.1 Mbps tile-mean MAE versus 17.1 Mbps for a simple mean baseline. The UI warns prominently that this is an illustration, not validated coverage.

## v13 — Collected Mobile Measurements

- Integrated the January–June 2026 Ookla mobile dataset as a separate default bandwidth evidence view, retaining original tile geometry, periods, speeds and counts.
- Added quarter and minimum-test filters, source/license attribution, sample-weighted summaries, boundary warnings and filtered CSV/GeoJSON downloads with notes.
- Kept low-sample exclusions gray and unsampled areas uncolored. No interpolation, technology inference or automatic scenario calibration is applied.
- Added local integrity/schema validation and unit, map, export and Streamlit regression checks. The reader does not require an API key, GEE, rasterio or a new runtime dependency.
- Preserved existing scenario, manual upload, hazard models and startup behavior. No model was retrained and no accuracy claim is made.

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
