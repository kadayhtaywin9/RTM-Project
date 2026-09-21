# GeoVision AI

**Team GeoVisionaries · University of Technology (Yatanarpon Cyber City)**

Telecom coverage planning, site suitability, multi-hazard exposure and SOS reporting for Yangon City, Hmawbi, Thanlyin and Kyauktan.

## Requirements

- Windows, macOS or Linux; **64-bit Python 3.11 or 3.12**.
- Internet access for installation, map imagery and live hazard sources.
- A browser with WebGL enabled. Phone location reporting requires HTTPS.

Extract the entire ZIP. Keep the project folders together. Demo hazard analysis uses packaged data and needs no credentials; maps still require internet access.

## Windows

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/), including the Python launcher.
2. Open the extracted `RTM-Project` folder and run `run_windows.bat`.
3. Open [the dashboard](http://localhost:8501).
4. Run `run_sos_windows.bat` in a second window; open [the SOS page](http://localhost:8000).

Both launchers use `.venv`, created by the dashboard launcher. Keep both windows open; press **Ctrl+C** in each to stop.

Manual setup in PowerShell, from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_runtime.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In a second terminal, from the same folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
```

If `py` is unavailable, use `python` for the first command after confirming `python --version` is 3.11 or 3.12.

## macOS and Linux

Install Python 3.11 or 3.12. On macOS, install the XGBoost OpenMP dependency with `brew install libomp` if using Homebrew. On Linux, install your distribution's matching Python `venv` package if needed. A missing `libgomp.so.1` requires its OpenMP runtime.

From the project folder:

```bash
bash run_mac_linux.sh
```

In a second terminal, from the same folder:

```bash
bash run_sos_mac_linux.sh
```

Open [the dashboard](http://localhost:8501) and [the SOS page](http://localhost:8000). Both scripts use `.venv`; the dashboard script installs dependencies on first use. Stop each service with **Ctrl+C**.

Manual alternative, using `python3.11` instead if applicable:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_runtime.py
.venv/bin/python -m streamlit run app.py
# Run in a second terminal:
.venv/bin/python -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
```

## Anaconda / Miniconda — any platform

Use a dedicated Conda environment instead of the `.venv` launchers. From the project folder:

```text
conda create -n geovision python=3.12 -y
conda activate geovision
python -m pip install -r requirements.txt
python scripts/check_runtime.py
python -m streamlit run app.py
```

In a second Anaconda Prompt or Conda-enabled terminal, enter the project folder:

```text
conda activate geovision
python -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
```

### Windows Anaconda Prompt — Earth Engine and Cesium

From the project folder, activate your environment and run:

```bat
conda activate geovision
pip install -r requirements.txt
earthengine authenticate
set "GEE_PROJECT_ID=nth-mantra-505115-m1"
python -c "import ee; ee.Initialize(project='nth-mantra-505115-m1'); print(ee.String('GeoVision GEE connected').getInfo())"
set "CESIUM_ION_TOKEN=YOUR_CESIUM_ION_TOKEN"
streamlit run app.py
```

Replace `geovision` with your environment name and `YOUR_CESIUM_ION_TOKEN` with your public client token. Use a Google account with access to `nth-mantra-505115-m1`, or replace the project ID in both commands with your own Earth Engine-enabled project.

Install requirements on first setup or after dependency changes. Authenticate on first use or when credentials expire. Run the two `set` commands again in each new Anaconda Prompt before starting the dashboard. These commands use Windows Command Prompt syntax, not PowerShell.

## Configuration

Local demonstration needs no configuration. Set environment variables before launching and restart after changes. The dashboard also reads `.streamlit/secrets.toml`: copy the example file and uncomment settings as needed. The SOS service reads environment variables, **not** Streamlit secrets.

- `CESIUM_ION_TOKEN`: optional public client token for satellite imagery and terrain. Without it, maps use OpenStreetMap imagery on a flat globe.
- `SOS_API_URL`: dashboard-to-SOS endpoint; default `http://127.0.0.1:8000`.
- `SOS_API_KEY`: shared key for reading incidents and changing status. Set the same value in both processes. Location submissions remain public.
- `SOS_DB_PATH`: SQLite database; default `sos_service/sos.db`. Use persistent storage when hosting.
- `SOS_ALLOWED_ORIGINS`: comma-separated SOS CORS origins; default `*`.
- `GEE_PROJECT_ID`: Earth Engine project for live rainfall.

Set a shared SOS key in **both** terminals before launching:

```powershell
# PowerShell
$env:SOS_API_KEY = "your-long-random-key"
```

```bat
REM Command Prompt / Anaconda Prompt
set "SOS_API_KEY=your-long-random-key"
```

```bash
# macOS / Linux
export SOS_API_KEY="your-long-random-key"
```

### Maps and live sources

- **Cesium:** create a [public client token](https://ion.cesium.com/tokens) with `assets:read` for World Imagery and World Terrain. Set `CESIUM_ION_TOKEN`; restrict allowed URLs to dashboard addresses. The token is visible in the browser: do not use administrative/write permissions. Update URL restrictions when tunnel addresses change. Dashboard maps use Cesium; the resident SOS page uses Leaflet.
- **Flood:** use an Earth Engine-enabled Google Cloud project. Set `GEE_PROJECT_ID` and run `python -c "import ee; ee.Authenticate()"` with the project environment's Python. Server credentials can use `GEE_SERVICE_ACCOUNT_JSON`. See [Earth Engine authentication](https://developers.google.com/earth-engine/guides/auth).
- **Earthquake:** USGS event catalog; no API key.
- **Cyclone:** JTWC/NRL ATCF products; no API key.
- **Compound:** component hazard outputs and tower isolation.

`Automatic` may use labeled historical fallbacks. `Live only` requires usable source evidence. `Demo data` uses demonstration inputs. Missing sources/events are reported separately from assessed scores.

## Review workflow

1. Select a study area and service radius.
2. Inspect coverage gaps, tower catchments and suggested sites.
3. In **Site checker**, select a map point or enter coordinates.
4. In **Hazard analysis**, select a hazard and source, then run. **Demo data** needs no credentials.
5. Submit a test location through the SOS page. Review, acknowledge or resolve it under **SOS Emergency**.

SOS searches towers within the project AOI: Yangon City, Hmawbi, Thanlyin and Kyauktan. Taikkyi and other townships outside this boundary are excluded. An SOS location outside the AOI still receives the nearest recorded tower inside it. The map displays the project boundary, location and selected tower. Sidebar selection does not narrow this search. Geographic proximity does not identify a phone's serving cell.

## Data and models

- Analysis: 36 townships, 194 Admin-4 areas and 8,029 tower-site proxies.
- Source catalog: 8,540 Yangon-region tower-site proxies and 14,873 cell records; SOS candidates are spatially restricted to the project AOI.
- Population: WorldPop 2020 baseline, approximately 7.22 million people in the analysis area.
- Coverage: geographic service radius; populated pixels are assigned to the nearest available site. Cached neighbours and spatial search support outage scenarios.
- Site suitability: XGBoost classifier using gap, population, rural priority, hazard safety and elevation.
- Hazards: flood, earthquake, cyclone and compound exposure scores. Training labels include planning proxies and authored rules; scores are not calibrated disaster probabilities or verified outage predictions. Model metadata records training scope and limitations.
- Impact: hypothetical tower unavailability, without RF, capacity, power or backhaul simulation.

This package includes application source, trained artifacts and runtime data. Training pipelines and evaluation tests are not included. Packaged datasets and model artifacts are preserved.

## Project layout

```text
app.py                    Streamlit dashboard
geoai_engine.py           Coverage and population calculations
site_ai.py / hazard_ai.py  Site and hazard inference
data/                     Runtime datasets and source connectors
models/                   Trained artifacts, wrappers and metadata
engine/                   Model orchestration and decisions
ui/                       Dashboard components and Cesium frontend
utils/                    Analysis, reports and SOS helpers
sos_service/              FastAPI service and resident web page
scripts/check_runtime.py  Dependency and import checks
```

## External access

Streamlit and SOS run as separate processes. Keep both running with persistent SOS storage. Use HTTPS for phone GPS and protect dashboard access before exposing incident locations. Set `SOS_API_KEY` before publishing SOS; an empty key leaves incident reads and status changes unrestricted.

For a temporary demonstration, install [cloudflared](https://developers.cloudflare.com/tunnel/get-started/) separately, start SOS, then run:

```text
cloudflared tunnel --url http://127.0.0.1:8000
```

Open the printed HTTPS URL on the phone. The local dashboard can keep using `http://127.0.0.1:8000`. If hosted elsewhere, set its `SOS_API_URL` to the reachable SOS address with the same API key.

[Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/) need no domain and use temporary URLs. The host and tunnel must remain running. Cloudflare forwards traffic; it does not run the Python services. `cloudflared` is not bundled.

## Troubleshooting and updates

- **Missing packages:** run `scripts/check_runtime.py` and install `requirements.txt` using the same Python that launches the app.
- **Windows Application Control / Device Guard:** have the administrator review the blocked executable or DLL and approve an allowed installation. Optional raster sampling can fall back to a labeled stored proxy if Rasterio cannot load.
- **macOS OpenMP error:** install `libomp`; see [XGBoost installation](https://xgboost.readthedocs.io/en/latest/install.html).
- **SOS unavailable:** check [service health](http://localhost:8000/health), `SOS_API_URL` and matching keys.
- **Phone GPS unavailable:** use HTTPS and allow location access. A phone's `localhost` refers to the phone.
- **Map blank:** check WebGL, network access and token permissions/URL restrictions.
- **Port in use:** stop the previous process, or pass `--server.port 8502` to Streamlit / `--port 8001` to Uvicorn. Update `SOS_API_URL` if the SOS port changes.

Before updating, stop both services and back up `.streamlit/secrets.toml` and `sos_service/sos.db` (or your custom database). Restore them into the new installation. No credentials, incident database or Python environment is included in this ZIP.
