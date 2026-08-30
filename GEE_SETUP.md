# Google Earth Engine setup for GeoVision Hazard AI

## What the live mode does

The `Hazard AI (GEE)` tab calls Google Earth Engine from the Streamlit backend. It retrieves:

- JAXA GSMaP v6 operational gauge-corrected hourly precipitation (`JAXA/GPM_L3/GSMaP/v6/operational`)
- 30-day rainfall accumulation used by Model 2
- 72-hour rainfall accumulation shown as current-event context
- SRTM elevation (`USGS/SRTMGL1_003`)
- SRTM-derived slope (`ee.Terrain.slope`)

Those values are sampled at the current tower coordinates, converted into Model-2 features, and passed directly to `models/hazard_flood_xgb.json`.

## 1. Google Cloud / Earth Engine

1. Create or choose a Google Cloud project.
2. Enable the Earth Engine API for that project.
3. Make sure the project is registered/eligible to use Earth Engine.
4. For local development, authenticate Earth Engine with your Google account or use a service account.
5. For a deployed Streamlit app, use a service account or another unattended credential method. Do not put a key file in GitHub.

## 2. Install packages

```bash
pip install -r requirements.txt
```

The updated requirements include `earthengine-api` and `google-auth`.

## 3A. Local user authentication

You can authenticate once in Python:

```python
import ee
ee.Authenticate()
ee.Initialize(project="YOUR_PROJECT_ID")
```

Then set the project ID before launching the dashboard:

Windows PowerShell:

```powershell
$env:GEE_PROJECT_ID="YOUR_PROJECT_ID"
streamlit run app.py
```

macOS/Linux:

```bash
export GEE_PROJECT_ID="YOUR_PROJECT_ID"
streamlit run app.py
```

## 3B. Streamlit deployment with service-account JSON

Use `.streamlit/secrets.toml.example` as the structure. Put the real values in Streamlit's Secrets UI or in a local `.streamlit/secrets.toml` that is excluded from source control.

The dashboard reads:

```toml
[gee]
project_id = "..."
service_account_json = '''{ ... full service account JSON ... }'''
```

## 4. Run

```bash
streamlit run app.py
```

Open **Hazard AI (GEE)** and choose:

- `Auto: GEE then local fallback` — recommended for a hackathon demo
- `Live Google Earth Engine only` — fail visibly if GEE is not configured
- `Local cached demo` — no GEE/network dependency

Then click **Run Hazard AI**.

## Important scientific limitation

Model 2 is currently an MVP exposure-score model. Its target is a pseudo-label made from rainfall stress, historic flood susceptibility, elevation and slope because the project does not yet contain date-and-location matched observed flood/no-flood labels. Do not call the output a calibrated flood probability.

For the next research version, replace `hazard_target_score` with observed event labels and retrain using:

```bash
python ml/prepare_hazard_training_data.py
python ml/train_hazard_xgboost.py
```


## Multi-hazard data sources

- Flood / Heavy Rain: Earth Engine `JAXA/GPM_L3/GSMaP/v6/operational` + `USGS/SRTMGL1_003`.
- Cyclone: Earth Engine `NOAA/IBTrACS/v4` observational best-track/archive context.
- Earthquake: USGS FDSN earthquake event service (no GEE credential required for that submodel).
- Compound: combines the three AI outputs.

IBTrACS is not a cyclone forecast feed; it is used as observational/archive context.
