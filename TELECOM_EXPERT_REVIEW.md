# GeoVision AI: critical telecom and AI review

Review date: 9 September 2026. Scope: the packaged Python application, model contracts, training scripts, metadata and deployment controls. References use repository-relative files and the line numbers inspected; subsequent edits can move them. This review does not establish the current condition of any operator's network. No credentials were inspected.

## Executive verdict

GeoVision is a useful GIS research prototype for exploring telecom exposure and geographic coverage scenarios. It is not yet a validated outage predictor, RF planning tool, measured connectivity monitor or deployment optimizer.

The hardest truth is that its AI mostly learns scoring formulas written by the project. A high validation score demonstrates that XGBoost reproduced those formulas; it does not demonstrate that the formulas describe real disasters, tower failures or customer service. Adding live APIs or a more polished map does not close that evidence gap.

The accompanying bandwidth map uses assumed Mbps per site and a user-selected active population share. Its outputs are hypothetical capacity-sharing scenarios. They are not measured download speeds, certified site capacity or operator performance. That feature does not correct the research and operational shortcomings below.

## Prioritized verified shortcomings

| Priority | Finding and practical consequence | Code evidence |
| --- | --- | --- |
| Critical | **The learning targets are authored proxies.** Site suitability labels reproduce the top quartile of an existing rule; flood labels directly combine the input factors. Real-world predictive value remains unproved. | `ml/prepare_training_data.py:123`, `ml/prepare_training_data.py:135`, `ml/prepare_hazard_training_data.py:59`; `models/MODEL_CARD.md`, `models/HAZARD_MODEL_CARD.md` |
| Critical | **Geographic proximity is treated as service eligibility.** Population moves to the nearest surviving site within a radius. RF propagation, antenna sectors, interference, operator compatibility, congestion, power and backhaul constraints do not determine whether that transfer can actually work. | `geoai_engine.py:133`, `ui/hazard_results.py:83` |
| Critical | **Public exception details can expose sensitive configuration.** The app renders the raw exception string. A credential parsing/SDK failure may contain the supplied credential object; safe report labels do not protect this separate UI path. | `app.py:2439`, `app.py:2447`, `data/gee_connector.py:193`; compare export sanitization at `utils/result_context.py:72` |
| High | **Training row counts overstate independent evidence if presented without context.** Flood's 128,464 rows are 8,029 sites repeated over 16 snapshots, with only 64 distinct district/date rainfall contexts in the packaged training data. Earthquake/cyclone repeat sites over six synthetic intensities. These are not observed disaster-case counts. | `models/hazard_model_metadata.json:11`, `ml/prepare_hazard_training_data.py:47`, `ml/train_multi_hazard_models.py:124` |
| High | **The tower inventory is a proxy.** Coordinates are deduplicated cell observations, not an operator-verified physical asset register. The packaged population baseline is 2020; neither current subscriber demand nor current operational site status follows from these inputs. | `data/metadata.json:17`, `data/metadata.json:29`, `data/metadata.json:32` |
| High | **An administrative key collision can attach the wrong population.** Two urban records in different townships share one Admin-4 code. The site checker keeps the first population record for each code, so the other township can receive that population; candidate grouping can also mix the locations. | `data/admin4_population_2020.csv:181`, `data/admin4_population_2020.csv:182`, `site_ai.py:62`, `site_ai.py:65` |
| High | **Engineering vulnerability and redundancy are inferred from weak proxies.** Radio generation and observed cell count are not structural strength, battery autonomy or independent power/backhaul paths. Nearby observations may also represent equipment on the same physical structure. | `data/preprocessing.py:52`, `data/preprocessing.py:61`; physical-site caveat at `data/metadata.json:17` |
| High | **Flood exposure is not flood hydraulics.** Four predictors use 30-day rainfall percentile, historical footprint, elevation and slope. The model does not predict water depth, arrival time or equipment inundation; live 24/72-hour rainfall and Dynamic World do not enter its trained feature vector. | `models/hazard_model_metadata.json:5`, `hazard_ai.py:284`, `utils/result_context.py:170` |
| High | **Historical and live rainfall are not demonstrably equivalent.** Training uses district-level WFP/CHIRPS history; runtime samples GSMaP at 10 km and compares its totals with that historical distribution. Cross-product/spatial-scale calibration was not found, and pooled seasonal percentiles need not represent unusual local rainfall. | `geoai_engine.py:71`, `ml/prepare_hazard_training_data.py:50`, `data/gee_connector.py:369`, `hazard_ai.py:193`, `hazard_ai.py:284`, `preprocess_population_disaster.py:203` |
| High | **Cyclone training is extremely limited.** The historical dataset contains one cyclone record. The live signal applies a hand-chosen wind/distance/forecast-lead formula; full IBTrACS retraining, site-level wind fields and observed tower fragility are not established. | `data/metadata.json:14`, `data/cyclone_api.py:488`, `models/MULTI_HAZARD_MODEL_CARD.md` |
| High | **Earthquake output is post-event heuristic exposure.** Magnitude, depth and distance enter an authored decay formula. It is not earthquake prediction or a calibrated site shaking/damage calculation. | `data/earthquake_api.py:89`, `data/earthquake_api.py:151` |
| High | **Compound scoring is not a physical joint-risk model.** Combining three proxy outputs with isolation does not establish joint event probability, common-cause failures, hazard sequences or restoration behavior. | `ml/train_multi_hazard_models.py:313`, `ml/train_multi_hazard_models.py:316`; `models/MULTI_HAZARD_MODEL_CARD.md` |
| High | **A saved live result does not automatically become stale in the UI.** The current-result signature checks hazard, area and mode but not elapsed time. Connector freshness checks only run when fetching/inference runs again. | `app.py:2409`, `app.py:2454`, `utils/result_context.py:59` |
| High | **Optional context can block required analysis.** The live flood path requires Dynamic World before returning environmental features, although land cover is not a model input. It also requires complete hourly rainfall and complete tower sampling; one missing observation can prevent the whole selected-area result. | `data/gee_connector.py:154`, `data/gee_connector.py:349`, `data/gee_connector.py:387`, `hazard_ai.py:255` |
| High | **Outside-area resilience is optimistic.** Unassessed towers are assumed available and eligible for rerouting. Nearby assets affected by the same flood, storm, power grid or fiber route can therefore appear as surviving alternatives. | `ui/hazard_results.py:83`, `ui/hazard_results.py:85`, `utils/scenario_report.py:65` |
| High | **Recommended sites are ranked points, not an optimized build plan.** Classifier ranking plus greedy spacing does not measure incremental service from the complete proposed portfolio or account for cost, feasibility and interference. A field named expected population coverage uses the surrounding administrative population rather than verified new customers served. | `site_ai.py:257`, `models/tower_model.py:23` |
| High | **Aggregate metrics conceal weak decisions.** The site model has only 194 examples and 49 rule-defined positives; 38 positives are in Kyauktan. The Hmawbi fold reports 97.4% accuracy but zero recall for its one positive. The selected threshold is also evaluated on the same out-of-fold predictions used to select it. | `models/xgboost_metrics.json:47`, `data/tower_training_data.csv` (audit grouping), `ml/train_xgboost.py:123`, `ml/train_xgboost.py:162` |
| Medium | **Uncertainty is described, not quantified.** The UI reports point scores, category thresholds and exact population totals without validated error ranges or rank-stability estimates. A manual threshold slider is useful sensitivity exploration, not a confidence interval. | `ui/hazard_results.py:41`, `ui/hazard_results.py:74`, `ui/hazard_results.py:93` |
| Medium | **Public service readiness is not demonstrated.** Caching and HTTP timeouts exist; app-level request limits, shared job admission, structured monitoring and a tested source-failure operating procedure were not found in the audited implementation. Deployment instructions request monitoring but do not implement it. | `utils/caching.py:10`, `data/earthquake_api.py:72`, `DEPLOYMENT.md:44` |
| Medium | **Exports do not fully reproduce a run.** Assumptions and source times are exported, but the report lacks an immutable input snapshot, exact artifact hashes and a code revision. GEE regression tests use mocked services and do not prove hosted IAM, latency or quota behavior. | `utils/scenario_report.py:59`, `ui/hazard_results.py:173`, `tests/test_gee_connector.py:173`, `tests/test_gee_connector.py:468` |

The issue with strict missing-data checks is availability, not that missing values should become zeros. Keep the validation; separate optional layers and return explicitly unassessed locations when the scientific contract permits partial analysis.

Missingness is not handled consistently across layers: common feature preparation fills several missing historical/terrain risk inputs with zero (`hazard_ai.py:149`). Unknown conditions can therefore resemble low exposure even though the live rainfall connector correctly rejects missing rainfall.

Geographic holdouts are a real improvement, but the compound validation is not fully nested: its outer-fold meta-model trains on in-sample child outputs, while final meta-model training uses cross-fitted child outputs (`models/MULTI_HAZARD_MODEL_CARD.md:43`). Neither arrangement creates independent observed outcomes.

## What the project needs to know

| Evidence needed | Minimum useful contents | What it would let you establish |
| --- | --- | --- |
| Verified asset inventory | Physical site and sector IDs, operator, technology, band, bandwidth, antenna height/azimuth/tilt, transmit power, coordinate accuracy, last verification time | Which assets exist and which subscribers they can serve |
| Service and traffic measurements | Timestamped site/sector utilization, active users, busy-hour throughput, latency, packet loss, accessibility and availability; geographically sampled field tests | Baseline capacity and customer service, rather than population-only demand |
| Dependency and resilience data | Fiber/microwave topology, aggregation/core dependencies, redundant paths, grid supply, battery autonomy, generator/fuel availability, structural condition | Failures that spread beyond an exposed tower and realistic surviving capacity |
| Matched disaster outcomes | Event ID, timestamp, local hazard intensity, outage start/end, affected sector/site, cause, repair action; observed non-failures as well as failures | Whether a hazard score predicts the outcome the operator actually cares about |
| Hazard observations | Historical event rainfall, verified flood extent/depth, ground shaking where available, cyclone tracks/intensity and uncertainty; local terrain/hydrology context | Hazard-specific backtesting and feature construction without future information |
| Build feasibility and response constraints | Land/roof access, lease and construction costs, transport access, power/backhaul connection costs, planning constraints, crews, repair time and critical facilities | Whether a recommended site or intervention is feasible and worth its cost |

More rows created by repeating towers are not a substitute for more independent events and verified outcomes. Acquire a modest, well-documented operator dataset before adding more model complexity.

## Uncertainty that must be reported

- **Inventory uncertainty:** physical-site identity, coordinate error, missing assets and operator membership.
- **Observation uncertainty:** rainfall resolution/error, event revision, latency, missing pixels and uneven measurement coverage.
- **Model uncertainty:** proxy-label bias, unsupported extrapolation, correlated inputs and the absence of calibrated failure probabilities.
- **Scenario uncertainty:** chosen radius, unavailable-score threshold, assumed bandwidth, active-user share and outside-area availability.
- **Outcome uncertainty:** congestion, upstream outages, power depletion, repair duration and changing population distribution.

Start with transparent scenario ranges and report how tower rankings and service estimates change across plausible assumptions. Label these as sensitivity ranges. Statistical confidence or prediction intervals require an appropriate method and independent evaluation; an arbitrary percentage around each score would add false assurance.

## Acceptance gates

1. **Public research release:** remove credential-bearing public errors; validate secret configuration safely; implement visible source ages and result expiry; distinguish missing/unassessed observations from zero; separate measured data from assumed bandwidth; document fallback behavior. Verify the hosted app with real read-only source requests and expected user concurrency. Test that an expired or unavailable source cannot produce a current-conditions claim.
2. **Operator-supported pilot:** obtain verified inventory and measurements for a bounded region. Check site identity, operator eligibility and field coverage; reproduce several documented outage cases including surviving sites. Agree on the decision and its cost before selecting evaluation metrics. The pilot passes only against criteria agreed in advance, not because a map looks plausible.
3. **Predictive evaluation:** train on observed outcomes; hold out entire events, future periods and geographic areas. Compare against the original rule, historical exposure and simple statistical baselines. Report event-level recall, false alarms, missed critical sites, ranking utility and calibration where probabilities are claimed. Fit normalization, imputation and feature selection inside training folds. Evaluate the final stacking procedure itself.
4. **Operational decision support:** add capacity/RF/dependency and restoration constraints justified by the operator data. Establish latency/availability targets, quota control, monitoring, source-failure procedures, audit manifests, access policy and human review. Demonstrate prospective benefit on new events before using recommendations for resource dispatch or capital approval.

No numeric model-performance target is prescribed here: the correct threshold depends on the specific decision, missed-outage cost, false-alarm cost and available response capacity.

## What is already strong

The project already labels outputs as research exposure scores, separates live/demo/fallback provenance, retains separate compound source clocks, and explains that SHAP reflects model behavior rather than physical causation (`ui/hazard_results.py:18`, `ui/hazard_results.py:139`, `utils/result_context.py:110`). Rainfall and cyclone validators reject important stale and malformed input cases. Population accounting enforces affected = rerouted + losing coverage, and export text is guarded against spreadsheet formulas (`utils/scenario_report.py:43`, `utils/scenario_report.py:79`). These are substantive engineering controls worth retaining.

As an external comparator, the [ITU Disaster Connectivity Map](https://dcm.itu.int/) distinguishes network infrastructure, mobile coverage and connectivity measurements such as download/upload speed and latency. It also acknowledges incomplete measurement coverage. The lesson is to join verified infrastructure and observed service evidence, while preserving their different meanings; reproducing the visual appearance alone would not provide that capability.

## Known issues left uncorrected by this review

This report changes no model, API connector, credential handling or coverage calculation. The shortcomings above remain open unless separately implemented and verified. In particular, proxy targets, absent observed-outage validation, missing operator/RF/backhaul/power constraints, raw public exception details and saved-result freshness are not fixed by adding a scenario map.

A defensible description today is: **“GeoVision is a GIS-based telecom resilience screening prototype with experimental learned scoring and explicit what-if coverage/capacity scenarios.”** Stronger claims need the acceptance evidence above.
