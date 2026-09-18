# GeoVision Resident SOS Service

A small resident-facing web app plus API. It stores coordinates, GPS accuracy, timestamps, incident status/source and the request's user-agent string. Treat the incident database as sensitive location data.

The v19 resident header shows Myanmar time (`Asia/Yangon`, UTC+06:30), seconds and the matching local date at the upper-right. The display uses the visitor's device clock, updates without reloading the page and does not need GPS permission. Server receipt timestamps remain UTC. A device with an incorrect clock will display an incorrect time.

## Run locally

From the GeoVision project root:

```bash
pip install -r requirements.txt
python -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` on the same computer. For a phone on another device, browser geolocation normally requires HTTPS; deploy this service behind HTTPS for the physical demo.

## API protection — required before public deployment

Set `SOS_API_KEY` on the SOS service. Put the same value in the dashboard configuration. Resident `POST /api/sos` stays public, while dashboard reads/status updates require the key.

With no key configured, reads and status changes are unauthenticated. Do not expose this default to the internet. A key alone is not production hardening: submission abuse protection, responder identities, retention rules and reliable dispatch are still missing. See [the v19 project review](../PROJECT_REVIEW_V19.md).

On Windows, use the same project Python environment for both apps; see [v19 launch instructions](../RELEASE_NOTES_V19.md). Restart the SOS service as well as Streamlit after upgrading. The clock belongs to this independent resident service, not to the Streamlit page.

Useful environment variables:

- `SOS_API_KEY`: protects dashboard read/update endpoints.
- `SOS_ALLOWED_ORIGINS`: comma-separated browser origins; `*` by default for demo use.
- `SOS_DB_PATH`: path to the SQLite database.
