# Google Earth Engine setup for GeoVision Hazard AI

## What the live mode does

The Flood / Heavy Rain workspace calls Google Earth Engine from the Streamlit backend. It retrieves:

- JAXA GSMaP v6 operational gauge-corrected hourly precipitation (`JAXA/GPM_L3/GSMaP/v6/operational`)
- 30-day rainfall accumulation used by Model 2
- 24-hour and 72-hour rainfall accumulations shown as current-event context
- SRTM elevation (`USGS/SRTMGL1_003`)
- SRTM-derived slope (`ee.Terrain.slope`)
- Dynamic World recent land-cover label and top-class confidence (`GOOGLE/DYNAMICWORLD/V1`)

Those values are sampled at current tower coordinates. The learned Flood inputs are 30-day rainfall percentile, historical flood score, elevation risk and slope risk. Dynamic World and the shorter rainfall windows are displayed context, not learned inputs. The Cyclone model does not consume live GEE features in this release.

Live rainfall requires a source timestamp no older than 48 hours (and at most one hour ahead of the app clock), all requested tower samples, finite nonnegative cumulative totals, and complete hourly inventories. Each pixel total is also masked unless all 24, 72 or 720 hourly observations are valid. The connector revalidates cached results. Failure raises in Live only and triggers a labelled local fallback in Automatic; missing rain is never silently treated as a live median-rainfall observation.

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

Open **Flood / Heavy Rain** and choose:

- `Automatic` — live source followed by a labelled local fallback
- `Live only` — fail visibly if GEE is not configured or the samples are invalid
- `Demo data` — no GEE/network dependency

Then click **Run analysis**.

## Important scientific limitation

Model 2 is currently an MVP exposure-score model. Its target is a pseudo-label made from rainfall stress, historic flood susceptibility, elevation and slope because the project does not yet contain date-and-location matched observed flood/no-flood labels. Do not call the output a calibrated flood probability.

For the next research version, replace `hazard_target_score` with observed event labels and retrain using:

```bash
python ml/prepare_hazard_training_data.py
python ml/train_hazard_xgboost.py
```


## Multi-hazard data sources

- Flood / Heavy Rain: Earth Engine `JAXA/GPM_L3/GSMaP/v6/operational` + `USGS/SRTMGL1_003` + `GOOGLE/DYNAMICWORLD/V1` environmental context.
- Cyclone current position and forecast: public JTWC operational products through the U.S. Naval Research Laboratory ATCF feed. No API key is required.
- Cyclone historical research/training: IBTrACS is the recommended archive for a future verified training-data pipeline; the bundled model has not yet completed that retraining.
- Earthquake: USGS FDSN earthquake event service (no GEE credential required for that submodel).
- Compound: combines the three AI outputs.

Earth Engine credentials are therefore needed for live flood/environmental sampling, but not for the JTWC or USGS public feeds. Auto mode falls back to packaged local data when a live source is unavailable.

Dynamic World attribution: This dataset is produced for the Dynamic World Project by Google in partnership with National Geographic Society and the World Resources Institute. It is licensed under CC BY 4.0 and contains modified Copernicus Sentinel data.
