# v19 — Classic Towers & Live SOS Clock

Based on `GeoVision_AI_v18_Tower_UI_SOS_Clock.zip`. Updated 14 September 2026.

## What changed

### Overview

The older compact blue-circle tower style is restored: 5-pixel points with partial transparency, site/radio/network hover details and the selected area's boundary. Gap markers and numbered suggested sites are back. Changing the study area now changes the displayed tower inventory, rather than always plotting all project towers.

For readability, more than 2,500 selected tower records are displayed as a fixed sample; a caption says so. This is a display limit, not a deletion or a new count of physical towers. Analysis still uses the selected inventory. Mapped records remain historical site proxies, not verified operational cells.

**Show bandwidth mask** switches away from the ordinary tower/gap/recommendation map to the existing evidence workspace. **Back to overview** restores it. Collected measurements remain the initial evidence view; full-AOI estimates, planning scenarios and manual uploads remain separate. The v18 distance-only speed illustration no longer replaces those views. No new speed measurements were collected and no models were retrained.

### Resident SOS page

A right-aligned clock shows `HH:mm:ss`, Myanmar's local date and `MMT · UTC+06:30`. It starts before map JavaScript/GPS initialization, updates every second while the browser permits and refreshes on returning to the page. It does not require location permission. The source is the visitor's device clock, not an independently synchronized time service. Server receipt timestamps remain UTC.

The five-second dashboard polling fragment is now attached to the incident feed, not the audio helper. API routes and incident storage format are unchanged. This remains a prototype, not a guaranteed emergency dispatch channel.

## Install and run on Windows

1. Extract the entire v19 ZIP into a **new folder**. Keep the old project, its secrets and any incident database as a backup. Do not copy an old `app.py` or `sos_service` over this release.
2. Open the extracted `GeoVision_AI_Multi_Hazard_Disaster_Model` folder and run `run_windows.bat`. It checks or creates a local `.venv` and starts the dashboard. Keep that window open.
3. Open a second PowerShell terminal in the **same extracted folder** and start the resident service with the same interpreter:

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn sos_service.app:app --host 127.0.0.1 --port 8000
```

4. Open the [local resident SOS page](http://localhost:8000) and Streamlit's displayed local URL. Stop any older process using these ports before starting the new version. Restart **both services** after upgrading; restarting Streamlit alone does not update the resident SOS page.

The older `run_sos_windows.bat` invokes the system's `python`; the explicit command above avoids accidentally using a different Conda/Python environment. If you already manage a working environment, activate it and run both services from that environment instead.

The local command deliberately listens only on this computer. For a phone demo, host the resident service behind HTTPS and configure the dashboard's `SOS_API_URL`. Streamlit hosting does not automatically deploy the separate FastAPI resident service. Set a nonempty `SOS_API_KEY` on the service and the matching value in the dashboard before any public deployment. Never put this key in resident JavaScript. Existing GEE secrets stay server-side and are not included in this archive.

If retaining real incident history, configure `SOS_DB_PATH` to a deliberately backed-up persistent database; do not accidentally start with an empty database and assume old incidents were resolved.

## Verification

- Python regression suite: **454 passed**, including four new map/API/polling tests. Existing dependency/deprecation warnings remain.
- `node scripts/test_sos_clock.mjs`: passed; checks immediate render, Myanmar midnight/year rollover, different device timezones and page-resume refresh.
- `python scripts/v19_ui_smoke_test.py`: passed; checks original map layers, Hmawbi area filtering, evidence toggle/back and retained SOS tab.
- Browser inspection: compact tower points and the upper-right live SOS clock rendered successfully. Location access was not granted and no real SOS was sent.
- Data and model files are preserved from v18. These tests check software behavior, not real-world telecom accuracy or emergency-response reliability.

Run tests after installing runtime and development requirements:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
node scripts/test_sos_clock.mjs
& ".\.venv\Scripts\python.exe" scripts/v19_ui_smoke_test.py
```

See [PROJECT_REVIEW_V19.md](PROJECT_REVIEW_V19.md) for prioritized remaining work.
