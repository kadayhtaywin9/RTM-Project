# GeoVision AI — Multi-Hazard Telecom Resilience Platform

**v21 — Live Assessment Gates & Restored Population Impact**, based on the delivered v20 project.

GeoVision is a research/planning dashboard for Yangon telecom resilience. It is not a calibrated disaster forecast, measured RF coverage system or emergency dispatch service.

## What changed in v21

- All four hazard workspaces distinguish usable live evidence from missing inputs, no-event source checks and historical planning.
- Missing, incomplete or unverified live evidence shows a gray **Not assessed** map. Scores, rankings and affected-population estimates remain unknown; historical-only values are not used as live results.
- A successful USGS query with zero qualifying earthquakes, or a verified JTWC index with no qualifying active forecast, shows **No matching events / no qualifying forecast**. It does not produce a background-only live score or imply safety.
- Compound Risk requires usable assessments from all three components. A no-event/missing component is not replaced by zero or a historical score. Inspect the individual hazards when a combined result is unavailable.
- **Population impact is visible by default again** for usable live assessments and explicitly labelled historical/demo scenarios. The main cards separately show people initially affected, people losing coverage and people rerouted; assumed tower outages and coverage after the scenario appear below them.
- Unknown values remain null in CSV/JSON exports. A failed run clears earlier scores and population totals.
- No models were retrained or rescaled. The 73-hour rainfall freshness policy, completeness checks, muted continuous colors, classic tower points, SOS receipt timestamps and resident clock are retained. Bandwidth masks remain removed.

Read [release notes and launch instructions](RELEASE_NOTES_V21.md). The [v20 model audit](MODEL_HONESTY_V20.md) and [its reproducible output](HAZARD_MODEL_AUDIT_V20.json) remain useful evidence of background bias; their old UI description is superseded by v21. No real-world accuracy or bias-free claim is made.

## AI systems

### Model 1 — Tower Recommendation AI

Ranks candidate areas using coverage gap, population, rural priority, terrain, and safety features. The stable output contract is:

- `tower_location`
- `priority_score`
- `expected_population_coverage`
- `recommendation_reason`

The packaged XGBoost classifier was trained on pseudo-labels generated from the earlier planning rule. Its output is a prioritization score, not proof that a site is technically or commercially feasible.

### Model 2 — Disaster Impact AI

The dashboard exposes four hazard workspaces:

1. Flood / Heavy Rain AI
2. Cyclone AI
3. Earthquake AI
4. Compound Risk AI

When usable inputs exist, each produces a tower score, risk category, recommended action and native XGBoost TreeSHAP explanation. The dashboard computes a hypothetical population impact from the selected service radius and tower-unavailability threshold. Without a usable assessment these outputs are unknown, not zero. Earthquake mode estimates tower impact after reported events; it does not predict earthquakes.

Population counts are not score-weighted population estimates: initially affected people are split into people rerouted to surviving towers and people losing geographic coverage. All surviving towers are eligible, including those beyond the ten-neighbor acceleration cache. Unassessed towers outside selected analysis areas are assumed available. Counts use the packaged 2020 population baseline, not current population or RF/capacity measurements.

All bundled hazard artifacts are MVP exposure models trained on pseudo-labels or synthetic event intensities. They are not calibrated event or tower-failure probabilities. Operational use requires verified event, outage, RF, asset-condition, power, and backhaul labels.

## Architecture

```text
data/                 External connectors and shared feature preparation
  gee_connector.py    Cached GEE rainfall, SRTM, and Dynamic World context
  earthquake_api.py   Cached USGS post-event ingestion
  cyclone_api.py      Keyless JTWC advisory/forecast ingestion and normalization
models/               Validated model loading, inference, and TreeSHAP
engine/               Hazard orchestration and action decisions
utils/                Caching and visualization helpers
ml/                   Offline dataset preparation and model training
app.py                Streamlit UI, maps, charts, and interaction
```

`site_ai.py` and `hazard_ai.py` remain as compatibility adapters so existing integrations continue to work during the incremental migration. See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow and limitations.

## Local setup

Python 3.11 or 3.12 is recommended.

On Windows, extract the **entire ZIP** and double-click `run_windows.bat`. The launcher locates compatible 64-bit Python, creates a project environment when absent, checks required runtime requirements/imports, and installs dependencies if needed. Rasterio is an optional runtime capability: a missing or blocked terrain reader gives a warning rather than preventing startup. First setup requires internet. Keep its terminal window open while using the app. Setup/server errors remain visible instead of disappearing; existing environments are never silently deleted. If a port is already in use, use the existing app or stop that server before launching another copy.

Manual setup is also available below. v8 requires Streamlit 1.63 or later within the supported major version, as specified in `requirements.txt`.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The dashboard remains usable in `Demo data` mode without GEE credentials.

### Windows: Application Control blocked a Rasterio DLL

If the traceback ends in `rasterio._base` with `An Application Control policy has blocked this file`, Windows refused to load a native terrain-library component. That is not a GEE authentication error. v10.1 removes the mandatory startup import. Overview, stored candidate ranking and the local hazard workspaces do not require this reader. Site checks use an existing stored area-candidate elevation proxy only when available, visibly labelled in the warning, metric, model-input table and downloaded report; this is not a fresh terrain sample or an accuracy improvement. If neither a valid sample nor stored terrain feature exists, site scoring is unavailable rather than substituting zero.

After installing this update, stop your old Streamlit process with Ctrl+C, open a terminal in the newly extracted project folder, and run `python -m streamlit run app.py`. No reinstall into Anaconda base or security-setting change is required for this workaround. The blocked terrain reader remains blocked. A failure is cached until process restart.

To investigate the blocked component, ask your device administrator to inspect **Event Viewer → Applications and Services Logs → Microsoft → Windows → CodeIntegrity → Operational**, including event 3077 and associated 3089 signature events. Use your organization's approved software/policy process; do not disable Windows protection. See [Microsoft's Application Control troubleshooting guidance](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting).

## Google Earth Engine

For local OAuth:

```bash
earthengine authenticate
```

Set the Cloud project before launch.

Windows PowerShell:

```powershell
$env:GEE_PROJECT_ID="your-google-cloud-project-id"
streamlit run app.py
```

macOS/Linux:

```bash
export GEE_PROJECT_ID="your-google-cloud-project-id"
streamlit run app.py
```

For Streamlit Cloud, use a service account in Streamlit Secrets. Never commit the real private key. Full instructions are in [GEE_SETUP.md](GEE_SETUP.md).

## JTWC cyclone feed

The live cyclone workspace reads public JTWC operational forecast products through the U.S. Naval Research Laboratory ATCF current-storm feed. It does not require an API key. The connector validates index/advisory timestamps; a stale index or a stale-only product set cannot establish that no storm is active. `Automatic` mode falls back to packaged historical background if the feed is unavailable, with current storm status explicitly unknown. `Live only` reports the failure without substituting demo data.

- Active-storm discovery: `https://science.nrlmry.navy.mil/atcf/index1.html`
- Forecast products: `https://science.nrlmry.navy.mil/atcf/docs/current_storms/{storm_id}.fst`
- Optional private-mirror overrides: `JTWC_INDEX_URL` and `JTWC_FST_BASE_URL`

JTWC supplies the current/forecast event signal only. The bundled Cyclone AI is still an MVP exposure model trained from the project's limited historical exposure proxy plus synthetic event intensities. A research/operational release still needs an IBTrACS-derived historical feature pipeline and verified cyclone/outage labels before retraining.

## Caching and performance

- SRTM elevation/slope use the permanent data cache.
- Dynamic World land cover uses a refreshed confidence-filtered recent composite.
- GSMaP rainfall is cached for 15 minutes and revalidated on return. The app requires a source timestamp no older than 73 hours (products over 48 hours are labelled delayed), complete hourly image inventories, and complete valid pixel observations for each 24/72/720-hour window.
- JTWC public forecast products use a short-lived cache and freshness validation.
- USGS events are cached for 5 minutes.
- XGBoost models and the hazard engine use the resource cache.
- GEE samples towers in one batched feature collection instead of making one request per tower.

Dynamic World is used under CC BY 4.0: the dataset is produced for the Dynamic World Project by Google in partnership with National Geographic Society and the World Resources Institute, and contains modified Copernicus Sentinel data.

The target is sub-10-second interaction after cache warm-up. First-run live GEE latency depends on Earth Engine quota and service availability; Auto mode falls back to packaged local data.

## Verification

```bash
pip install -r requirements-dev.txt
python scripts/smoke_test.py
python scripts/app_smoke_test.py
pytest -q
```

The smoke test loads all packaged models, scores candidate sites, and runs all four hazard engines in local mode.

The hazard workspace visibly distinguishes live, mixed, historical/demo, fallback and unknown input provenance. Live inputs do not establish model accuracy. Downloadable CSV results and JSON scenario notes retain source status/times, score scales, service radius, failure-threshold assumptions and the 2020 population reference. Source/product time is kept separate from analysis completion time; compound inputs retain separate timestamps.

The updated earthquake/cyclone/compound artifacts and metadata must be deployed together. Fixed training-set normalizers keep an identical tower's features independent of the selected batch. Compound training uses cross-fitted upstream predictions and township-held-out validation; the metrics still measure proxy-formula reproduction, not observed disaster skill.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Community Cloud configuration, secrets, health checks, and release guidance.

## Resident SOS prototype

This release includes a separate resident-facing SOS service in `sos_service/` and an **SOS Emergency** workspace in the Streamlit dashboard. See `SOS_FEATURE.md` for architecture, setup, limitations and deployment notes.

Run the SOS service with `run_sos_windows.bat` on Windows or `./run_sos_mac_linux.sh` on macOS/Linux, then run the normal dashboard separately.
