# GeoVision AI v10 — Bandwidth Evidence Update

## What was misleading in v9

The red blanket was the result of illustrative capacity/demand assumptions plus a geographic radius. It did not demonstrate poor actual service. Population outside the radius was assigned a scenario zero, and 1 km display-cell averages mixed it with supported population. The headline also counted this unassessed population as below a throughput target.

## Changes

- **Gray unassessed locations.** Throughput summaries now condition on population within the selected network's assumed radius. Cells containing any unsupported population are gray, with their supported share in hover. Unknown locations are never described as zero measured Mbps. The caption reports eligible population below target and unassessed population separately.
- **500 m display cells by default**, with a 1 km option. Both polygons and their contributing population centres must be inside the selected study boundary. No smoothing creates values between observations; display resolution is not RF/model accuracy.
- **Mapped operator selection.** The scenario recalculates nearest-site assignment using only sites observed for the chosen operator. The full project's population remains in the demand calculation before display filtering. Operator labels come from the historical inventory; they do not confirm current ownership or operating status. Technologies/sectors remain aggregated, and all-operator pooling is still available as an explicitly labelled scenario.
- **Assumption sensitivity.** A table compares half-capacity/double-demand, selected inputs and double-capacity/half-demand cases. These are deliberate stress cases, not statistical confidence intervals or calibrated operator bounds.
- **A separate measured-download view.** Upload real speed-test readings using the empty CSV template. This path never substitutes assumed capacity for missing measurements. It filters by operator, area, time window and location accuracy, then displays sample median, count, range and latest timestamp in 500 m bins. No interpolation fills unmeasured locations.
- **Preserved controls.** Scenario capacity, active share, target, operator and display resolution persist across evidence-mode changes in the session. The existing Overview button and return-to-map interaction remain.

## Measurement CSV

Required columns:

```text
latitude,longitude,download_mbps,measured_at,operator,location_accuracy_m
```

- Maximum 5 MB and 50,000 rows; UTF-8 CSV. The app offers a header-only template, with no fabricated sample measurements.
- Use WGS84 latitude/longitude and documented GPS/field location accuracy. IP geolocation is not appropriate for site-level conclusions. The default 500 m bins require stated location error no greater than 250 m.
- Timestamps must include a timezone (UTC `Z` or an offset such as `+06:30`). Invalid or more-than-five-minutes-future readings fail validation. Choose a 1, 7, 30 or 90 day window, default seven days.
- Coordinates, speed and location accuracy must be finite. Missing readings are rejected; a supplied nonnegative zero speed is retained as an observation. Operator names are required, and different operators are not pooled in this mode.
- Exact duplicate rows are removed. Out-of-window, imprecise and out-of-area counts are reported; a row may fail more than one filter.
- A colored bin means there are accepted samples there, not that its entire area has uniform coverage. One sample is not statistically representative. Supplied measurements have not been independently verified. Devices, test methods, traffic conditions and sampling bias still affect interpretation.

## Limits unchanged

The planning scenario is still based on 2020 population, a uniform assumed per-site budget, assumed active users and geographic distance. There is no verified spectrum, RF, antenna, traffic, power or shared-backhaul model. Site observations may duplicate physical infrastructure. No new operator performance dataset is bundled, and no AI was retrained. Improvements to evidence handling do not establish a numerical increase in prediction accuracy.

The earlier telecom review remains applicable to the hazard models and wider project. In particular, the duplicate administrative-code lookup, raw public hazard exception details, saved-result freshness and missing observed-outage validation are separate open issues. GEE authentication is unchanged; neither bandwidth mode requires GEE.

## Validation and environment limitation

On 9 September 2026, 65 bandwidth regression tests passed without warnings: the 43 existing calculation tests plus 22 evidence/operator/measurement tests. Scoped Ruff checks passed. The isolated Streamlit bandwidth smoke test passed with the actual project grid, operator selection, unknown-area layer, mode transitions and persistent controls; a synthetic upload fixture verified the complete CSV-to-measured-map rendering path. Synthetic tests are not supplied as project measurements.

The full application's smoke test could not complete because this Windows environment's Application Control policy blocked an existing `rasterio` DLL, including when retried outside the sandbox. No security policy was changed. This release therefore does not claim a successful full-app rerun in this environment. The earlier v9 full-suite results are historical, not current v10 validation.

## Run and update

Extract the ZIP into a new folder. Install `requirements.txt` in a compatible Python environment and run `python -m streamlit run app.py`, or use the existing launcher. Restart the Streamlit process after replacing files. No new dependency was added.

The v10 runtime changes are `app.py`, `engine/bandwidth_scenario.py`, `engine/bandwidth_evidence.py` and `ui/bandwidth_evidence_map.py`. The v9 renderer remains only for regression comparison and is not imported by the app. Upload/commit the updated project to the repository used by Streamlit Cloud and reboot there to update a hosted dashboard. Never include credentials in the repository or ZIP.
