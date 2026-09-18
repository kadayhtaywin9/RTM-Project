# v20 — Source Transparency & SOS Receipt Times

Updated 15 September 2026. Based on the delivered v19 project, with its classic tower-point Overview and resident SOS clock retained.

## Requested changes

1. **Bandwidth masks removed.** No bandwidth button, measurement mask, full-AOI interpolation or capacity scenario remains in the application. Associated UI/calculation modules and obsolete feature tests/scripts were removed from this release only. Raw source datasets and previous release documentation are preserved for research/recovery; your v19 ZIP is unchanged.
2. **SOS receipt time.** Incoming rows and selected incidents show the server's receipt date, time and seconds in Myanmar time (UTC+06:30). Missing or timezone-ambiguous records show “Receipt time unavailable”, never the current time.
3. **External data inventory.** A visible section shows what was used, its units and its timestamps. Compound results show each source separately.
4. **73-hour rainfall freshness tolerance.** The latest source image can be at most 73 hours old; products over 48 hours old are marked delayed. Completeness checks and 24/72/720-hour accumulation windows remain intact.
5. **Softer gradual map colors.** A continuous muted green–yellow–rose gradient replaces abrupt categorical color jumps. Color changes do not reduce scores or change thresholds.
6. **More honest result interpretation.** Large hypothetical outage/population totals are hidden unless the user explicitly enables the scenario. Earthquake/cyclone models show a zero-event reference alongside their current-input scores. Real-world accuracy, calibrated uncertainty and absence of bias remain unproven.

## Understanding the source summaries

| Hazard | What the displayed count means | Time information |
| --- | --- | --- |
| Flood | Unique hourly GSMaP source images, three overlapping aggregation periods and sampled tower rows, all separate units | Latest image, full interval, retrieval time and current product age |
| Earthquake | Catalog records returned, earthquakes accepted, ignored records and event magnitudes/depths | Actual query start/end, configured days, each event time and retrieval time |
| Cyclone | Forecast files retrieved, fresh active storms and current/forecast track points | Advisory interval, age limit, source-check time and forecast horizon |
| Compound | The three source records above; no misleading summed “data” count | Each component keeps its own clocks |

A complete 30-day rainfall inventory has 720 unique hourly images. The 24- and 72-hour totals overlap that inventory; they are not another 96 independent images. GEE computes the aggregates remotely and samples tower locations; this is not a download of 720 raster files to the dashboard. The approximately 10 km sampling scale means many tower rows may share weather pixels. SRTM and Dynamic World imagery counts are not instrumented and are not invented.

The packaged Flood model directly uses the 30-day rainfall percentile, flood history, elevation risk and slope risk. Its 24-hour/72-hour totals and land-cover output are contextual. Raising the freshness limit does not retrain the model or improve its accuracy.

USGS still uses the existing 30-day, minimum-magnitude-3 query by default. The UI reports the actual window, not example values of three or four days. Its strongest magnitude/depth/distance signal over that window has no time decay, so old events can influence the score. Magnitude is not local shaking intensity or an outage observation.

API responses may be cached. A recorded retrieval time stays the original fetch time; a new analysis timestamp does not make old observations fresh. Demo/fallback results explicitly say no live external records were used. Unrecorded counts are “Not recorded”, not zero.

## Windows launch

Extract the full ZIP into a new folder. Do not merge with old source files: merging could leave removed modules behind. Keep old secrets and incident databases backed up; neither credentials nor real incident data is included here.

Run `run_windows.bat` from the extracted project folder to set up/start Streamlit. For SOS, open a second PowerShell terminal in that same folder:

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn sos_service.app:app --host 127.0.0.1 --port 8000
```

Keep both terminals open. The resident page is [localhost:8000](http://localhost:8000); Streamlit prints its dashboard URL. Stop older processes on the same ports. Restart the dashboard after upgrading; restart the separate SOS service if you are replacing its files too. Do not use `python app.py`.

For public hosting, deploy the SOS service separately behind HTTPS, configure `SOS_API_URL`, and set matching nonempty `SOS_API_KEY` values on service/dashboard. The default empty key leaves incident reads and status updates open. The service remains a prototype without guaranteed delivery or dispatch. If retaining incident history, deliberately configure a backed-up persistent `SOS_DB_PATH`.

## Verification

- **264 Python tests passed.** This release removes 210 tests belonging to the deleted bandwidth feature and adds 20 focused v20 cases. It retains the other v19 regressions; the lower total is due to feature removal, not skipped failures.
- Full Streamlit app test passed for all four local hazard workspaces, optional scenario on/off, unchanged inference scores, study-area filtering, no bandwidth buttons, SOS receipt time and stale-result hiding.
- Local browser checks confirmed the muted continuous map and receipt-time layout using a clearly fictional, read-only SOS feed. No real SOS was submitted.
- Resident-clock JavaScript tests passed, including second-by-second rendering, Myanmar midnight/year rollover and timezone independence. Dependency/deprecation warnings remain.
- Data files and saved model artifacts were not changed. External API success on the user's deployment and real-world hazard accuracy were not verified.

See [MODEL_HONESTY_V20.md](MODEL_HONESTY_V20.md) and [reproducible audit output](HAZARD_MODEL_AUDIT_V20.json).
