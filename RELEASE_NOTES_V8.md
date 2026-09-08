# GeoVision AI v8 — Clear Results & UI Update

## What changed

- A map-first hazard workspace, with a compact summary of what-if coverage impacts, three sites to review first and a sortable review queue. The university/team network header and OpenStreetMap design are preserved.
- Clear live, mixed, demo/historical, fallback or unverified evidence status. Training counts and technical charts no longer dominate the result view.
- Exposure is labelled as a score, not a failure probability. Towers are **assumed unavailable in a scenario**, not reported as confirmed failures. Recommendations are engineering-review guidance conditional on verified alerts and site evidence.
- Compound models reject missing/duplicate tower inputs and invalid child scores. Decision outputs reject invalid scores and use consistent score-category thresholds.
- CSV exports include all assessed towers, not just visible rows. Companion JSON notes include per-source timestamps/status, selected areas, radius, assumed-unavailability threshold, population reference, model features and scientific limitations. Population totals and per-tower accounting are checked before export. Formula-like external text is escaped for spreadsheet safety.
- A Windows launcher that locates compatible Python, uses its own project directory, checks dependencies and keeps setup/startup failures visible. Existing environments are preserved.

## Run on Windows

1. Extract this ZIP into a **new folder**, keeping v7 for rollback. Do not run from inside the compressed archive.
2. Install 64-bit Python 3.12 or 3.11 if it is not already installed.
3. Double-click `run_windows.bat`. Allow the initial dependency installation to finish; it requires internet.
4. Open the Local URL printed by Streamlit and **keep the terminal open**. Closing it stops the dashboard.

If an existing `.venv` is broken or incompatible, the launcher will explain the problem and preserve it. Repair or rename that environment yourself before retrying; nothing is silently deleted. Do not copy a virtual environment from another computer.

Manual launch from the extracted project folder, after dependency setup:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Demo data works without external credentials. Live flood analysis still requires your Earth Engine project/authentication. Public JTWC and USGS availability is not guaranteed by this application.

## Scientific boundary

The packaged model artifacts are unchanged from v7. This update improves data integrity, interpretation and usability; it does **not** claim better measured forecasting accuracy. Models still use proxy/synthetic labels. Full IBTrACS historical training and validation against observed tower outages are not included. Cyclone does not yet consume live GEE rainfall or Dynamic World; Dynamic World remains contextual information in Flood.

Population uses the packaged 2020 baseline and geographic nearest-tower/radius screening. It is not an observed count of affected people or an RF, capacity, power or backhaul model. Unassessed towers outside selected areas are assumed available. A source observation timestamp differs from the analysis completion time; compound sources retain their separate clocks.

## Checks

Release validation on 2026-09-07: 216 regression tests passed, including startup stubs, source provenance, invalid compound inputs and export accounting. All four hazard workspaces passed Streamlit smoke tests at approximately 4–5 seconds per workspace in the local test environment. Nine existing pandas/NumPy timedelta deprecation warnings remain; they did not fail the tests. Installed dependency checks and scoped Ruff checks passed.

After an upgrade, restart the Streamlit process to discard old imported modules and cached engine objects; browser refresh alone may not be sufficient.

Run `pytest -q`, `python scripts/smoke_test.py` and `python scripts/app_smoke_test.py` after installing `requirements-dev.txt`. `python scripts/check_runtime.py` verifies installed requirements/imports without network access. Offline source fixtures test malformed/stale/mixed data. Authenticated live GEE and real-world outage validation remain deployment/research checks, not completed claims.
