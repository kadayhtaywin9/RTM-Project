# v21 — Live Assessment Gates & Restored Population Impact

15 September 2026. This release updates v20; earlier ZIPs are unchanged.

## Behavior by mode and evidence

| Situation | Hazard display | Population-impact display |
| --- | --- | --- |
| Usable live inputs | Event/rainfall-informed planning scores; historical/site influences remain disclosed | Hypothetical outage scenario, visible by default |
| Live API failure, incomplete imagery, missing provenance or stale required products | Gray: Not assessed; no risk ranking | Not assessed, never a fabricated zero |
| Successful USGS query: no qualifying earthquakes | Source-scoped no-event notice; gray availability map; no background-only score | Not assessed |
| Verified JTWC index: no qualifying active forecast | Source-scoped no-forecast notice; gray availability map; no background-only score | Not assessed |
| Demo data / Automatic historical fallback | Explicitly historical planning; not current-event evidence | Hypothetical historical scenario, visible by default |
| Compound with any unusable/no-event component | Combined assessment unavailable; inspect individual component statuses | Combined impact not assessed |

No matching catalog event/forecast is not a guarantee of safety. Missing data is not the same as a successful zero-record query. The green/yellow/rose gradient is used only for numeric planning scores; gray is used for unknown assessments.

## Restored affected-population function

Run a model with usable evidence, or explicitly choose Demo data for a historical experiment. **Show hypothetical tower-outage scenario** starts checked. Set the score threshold and population service radius to examine the consequences IF those towers became unavailable.

- **People initially affected:** baseline population assigned to the towers assumed unavailable in the scenario.
- **People rerouted:** the initially affected population for which the simulation finds another surviving tower within the chosen radius.
- **People losing coverage:** the remaining initially affected population without a surviving tower within that radius.
- **Assumed unavailable:** number of assessed towers crossing the chosen score threshold; not observed failed towers.
- **Coverage after scenario:** modeled geographic coverage of the selected baseline population after those assumed outages.

The accounting is: initially affected = rerouted + losing coverage. Do not add all three counts as separate affected groups. Values are based on the packaged 2020 population data, not a current census or observed victims. Surviving towers are not checked for actual RF, capacity, power or backhaul availability. Towers outside the selected area are assumed available. These are engineering what-if results, not emergency impact reports.

The scenario can still be switched off. When live evidence is not assessable, the panel remains visible with unknown values; it does not calculate an unsupported outage mask. Changing the threshold never changes model predictions.

## Model-specific safeguards

- **Flood:** retain the existing exact hourly completeness, pixel and finite-rainfall checks. The latest image may be at most 73 hours old; products over 48 hours are delayed. v21 also requires recorded image counts, sample coverage and retrieval/product timestamps before presenting a live assessment. Zero measured rainfall is data; missing rainfall is not zero. The trained model still uses a 30-day rainfall percentile plus historical/site features, not observed inundation.
- **Earthquake:** no-event checks skip event-model inference. Usable catalog evidence requires consistent record counts, a recorded query period, and event times within that period. Reaching the 2,000-record limit is treated as potentially incomplete and blocks assessment. Valid events in the 30-day query may be old; this remains post-event screening without time decay, not ongoing-damage detection or earthquake prediction.
- **Cyclone:** no-active-forecast checks skip model inference. A zero-record result requires a verifiable source-index update timestamp; an unverified or stale index is unknown. Advisory/index age checks remain enforced. Zero forecast products do not produce a displayed zero wind speed or zero background-risk score.
- **Compound:** the saved meta-model expects all three component scores. It was not trained for missing/no-event components, so v21 does not inject zero or a historical-only substitute. This conservative rule can make live Compound unavailable even while an individual hazard can be assessed. Replacing this rule requires a separately validated partial-input model, not cosmetic changes.

## Runtime/export contract

The public `run_hazard_ai` and `HazardEngine.run` results now carry `assessment` metadata: `status`, `label`, `reason`, `score_allowed`, and component states for Compound. Internal model prediction helpers remain available for offline experiments.

Integrations must check `run_info["assessment"]["score_allowed"]` before using scores. Unsupported results have null/NaN scores, nullable unavailable flags and unknown population fields; old recommendations/background scores are not retained in those results. Do not fill these nulls with zero.

CSV exports preserve unknown fields as blanks. JSON availability reports use null population totals, zero assessed towers and a separate listed-tower count. Scenario exports require a usable assessment and verify population conservation. Source-check time and event/product time remain separate. Re-running after a provider failure replaces stale prior outputs.

## Install and run

Extract the entire ZIP to a new folder; do not merge it over an older project. Run `run_windows.bat` and keep its terminal open. Stop the previous dashboard process first if it occupies the port. The new panel defaults apply in the new app session.

For the separate SOS service, follow `SOS_FEATURE.md`; its API, server receipt timestamps and resident clock are unchanged. Back up and deliberately configure your existing persistent SOS database before moving deployments. No real credentials, incident database, Python environment or visual-test fixtures are bundled.

## Verification and limits

- 320 Python tests passed: 264 existing tests plus 56 v21 cases. Dependency/deprecation warnings remain.
- Full app checks passed for all four hazards: default impact display, unchanged scores when toggling scenarios, local/demo and synthetic live evidence, no-event/unknown maps, failed-run clearing, Overview filtering, site checker and SOS receipt time.
- Resident-clock tests and focused changed-Python lint passed.
- Browser QA used historical data and synthetic no-event fixtures only. It verified full-width population counts and a gray unknown map. No provider call or real SOS submission was made in that visual test.
- Saved model files and datasets are unchanged. This is a correctness/transparency upgrade, not retraining, calibration, proven absence of bias or verified real-world accuracy. External API availability on your deployment remains unverified.

The earlier `MODEL_HONESTY_V20.md` explains why background scores can remain high. Its old UI behavior is superseded by this release.
