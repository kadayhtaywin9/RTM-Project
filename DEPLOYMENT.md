# Streamlit Cloud Deployment

## Prepare the repository

1. Deploy the complete extracted project, including `app.py`, `requirements.txt`, `data/`, `models/`, `engine/`, `utils/`, and the v8 `ui/` package. The compatibility modules in the project root are also required.
2. Do not commit `.streamlit/secrets.toml` or any service-account JSON file.
3. Run `python scripts/smoke_test.py` and `pytest -q` before release.
4. Choose `app.py` as the Streamlit entry point.

## Earth Engine service account

Create a Google Cloud service account, enable the Earth Engine API for its project, register/authorize it for Earth Engine, and grant only the permissions needed to read the referenced public datasets.

In Streamlit Cloud, open **App settings → Secrets** and add:

```toml
[gee]
project_id = "your-google-cloud-project-id"
service_account_json = '''
{
  "type": "service_account",
  "project_id": "your-google-cloud-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@your-google-cloud-project-id.iam.gserviceaccount.com",
  "client_id": "...",
  "token_uri": "https://oauth2.googleapis.com/token"
}
'''
```

Use `.streamlit/secrets.toml.example` as the non-secret template.

## Runtime behavior

- `Automatic` attempts the selected model's live sources and falls back to labelled packaged local inputs.
- `Live only` surfaces connector failures and invalid/stale data instead of silently falling back.
- `Demo data` is deterministic and useful for health checks. Cyclone demo is historical background, not a current event.
- Deploy all v7 code, model artifacts and metadata together. Do not mix the fixed-normalization runtime with older metadata.
- Changing data source, hazard or area hides the previous analysis until it is run again. Radius and failure-threshold changes recompute coverage from the current scores.
- Restart the Streamlit process after upgrading; a browser refresh alone may leave old imported modules or cached engine objects in a long-running process.
- v8's Windows launcher checks the full runtime with `scripts/check_runtime.py`. Streamlit must satisfy the updated minimum in `requirements.txt`.

## Release checks

- Confirm all four local hazard modes return scores for the expected tower count.
- Confirm model metadata feature order matches each artifact.
- Confirm no private key appears in Git history or logs.
- Warm the static GEE cache and check cached dashboard interactions are below the 10-second target.
- Review model/data dates and limitations shown in the UI.
- Monitor GEE quota, USGS/JTWC public-feed availability, exceptions, source freshness, and latency.
