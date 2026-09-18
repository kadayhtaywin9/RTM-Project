# v13 — Collected Mobile Measurements

## Open the new view

1. Extract the complete v13 project into a new folder. Keep your previous version as a backup.
2. Run `run_windows.bat`, or activate your working environment and run `python -m streamlit run app.py` from the project folder.
3. Open **Overview → Show bandwidth mask → Collected mobile measurements**.
4. Choose a quarter and minimum test count. Choose **1** to inspect all available tiles, including single-test observations. The default is **3**, a screening choice rather than a statistical confidence guarantee.
5. Hover a tile for its source mean download/upload, latency, tests, device count and quarterly period. Expand **Measurement records and download** for a filtered data package with source notes.

On Streamlit hosting, upload the complete project including `data/mobile_performance`, `engine/mobile_performance.py` and `ui/collected_mobile_map.py`, then reboot the app. The collected-data view requires no API key or additional dependency. Online basemap tiles still require internet access.

## Included evidence

650 tile-quarter records across 479 distinct tiles, summarizing 2,498 participating mobile speed tests from January–June 2026. With all source tiles included, the test-weighted means are 34.737 Mbps in Q1 and 34.976 Mbps in Q2. Area and sample filters can change these summaries. These are not citywide averages or tower capacities.

301 records contain one test. Operator, radio generation, serving tower, per-test timestamps, numeric GPS accuracy and signal strength are unavailable. Whole tiles are retained at boundaries; no interpolation or artificial zero filling is applied.

## Preserved behavior

- **Planning scenario** retains its load sharing, technology assumptions and color controls. It has not been calibrated to the collected speeds.
- **Measured download speeds** retains the individual-test CSV workflow, including real technology, timestamp and location-accuracy requirements.
- Existing hazard/site models, ordinary overview map, university/team header and optional-terrain startup behavior remain unchanged.
- Missing/corrupted collected data produces an explicit error in that view, never an unlabelled scenario fallback.

This release integrates evidence; it does not train a new AI model. See [measurement methodology](MOBILE_MEASUREMENTS.md) for the appropriate modeling scope and source license.
