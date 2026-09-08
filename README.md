# GeoVision AI — Multi-Hazard Telecom Resilience Platform

GeoVision AI is a Streamlit planning platform for Yangon telecom coverage and disaster resilience. It preserves the original dashboard while separating external data access, model serving, orchestration, decisions, and UI concerns.

This is **v8 — Clear Results & UI Update**. It retains the clean light dashboard, OpenStreetMap maps, and university/team network header, with a map-first hazard workspace, visible evidence labels and scenario-aware exports. See [RELEASE_NOTES_V8.md](RELEASE_NOTES_V8.md) for changes and remaining limitations.

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

Each produces a tower score, risk category, recommended action, and native XGBoost TreeSHAP explanation. The dashboard then computes population impact from the selected service radius and tower-unavailability threshold. Earthquake mode estimates tower impact after reported events; it does not predict earthquakes.

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

On Windows, extract the **entire ZIP** and double-click `run_windows.bat`. The launcher locates compatible 64-bit Python, creates a project environment when absent, checks all runtime requirements/imports, and installs dependencies if needed. First setup requires internet. Keep its terminal window open while using the app. Setup/server errors remain visible instead of disappearing; existing environments are never silently deleted. If a port is already in use, use the existing app or stop that server before launching another copy.

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
- GSMaP rainfall is cached for 15 minutes and revalidated on return. The app requires a source timestamp no older than 48 hours, complete hourly image inventories, and complete valid pixel observations for each 24/72/720-hour window.
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
