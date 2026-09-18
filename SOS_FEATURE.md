# GeoVision SOS Prototype

## What was added

### Resident web application

`/sos_service` is an independent FastAPI application. Its root page is a deliberately minimal phone UI:

- live OpenStreetMap/Leaflet map;
- browser GPS position and accuracy circle;
- upper-right live Myanmar-time clock, local date and timezone label (v19);
- one `SEND SOS` button;
- automatic payload: latitude, longitude, GPS accuracy and timestamps.

The resident does not need to type a name, phone number or message for this prototype.

### SOS API

- `POST /api/sos` accepts a resident SOS.
- `GET /api/sos` returns recent incidents to the GeoVision dashboard.
- `PATCH /api/sos/{id}/status` updates an incident to `ACKNOWLEDGED` or `RESOLVED`.
- SQLite stores incidents for the demo.
- `SOS_API_KEY` protects dashboard read/update operations when configured; without it these endpoints are open. A nonempty key is required before public deployment. Resident submission remains open.

### GeoVision dashboard

A new **SOS Emergency** tab:

- polls the SOS service every 5 seconds;
- shows a Streamlit toast when a new SOS arrives;
- resolves the SOS point to an ADM3 township using `yangon_admin3.geojson`;
- calculates the nearest mapped tower using great-circle distance against `tower_sites_yangon_all.csv`;
- displays incident coordinates, GPS accuracy, township, nearest tower, network/radio metadata and distance;
- `Locate` focuses a map on the victim and nearest tower and draws a link between them;
- supports simple `ACKNOWLEDGED` / `RESOLVED` state changes.
- shows the server receipt date and time in Myanmar time in each visible incident row and the selected incident (v20). The receipt clock is separate from device location time and responder acknowledgement.

## Run the prototype

Terminal 1 — SOS service:

```bash
python -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
```

Terminal 2 — GeoVision dashboard:

```bash
streamlit run app.py
```

Desktop local test:

- resident app: `http://localhost:8000`
- dashboard: Streamlit's displayed URL, normally `http://localhost:8501`

### Phone test

A phone browser normally requires **HTTPS** before it will expose precise browser geolocation. For the physical hackathon, deploy the SOS service to an HTTPS host and set the dashboard's `SOS_API_URL` to that public service.

## Production limitations

This is a hackathon prototype, not an emergency-services platform. It currently does **not** provide:

- iPhone Emergency SOS via satellite integration;
- SMS fallback or telecom-operator integration;
- offline/mesh/LoRa delivery;
- guaranteed delivery, dispatch or emergency-service acknowledgement;
- identity/phone-number verification;
- abuse/spam prevention or robust rate limiting;
- redundant databases, queues or high-availability infrastructure;
- encrypted application-level payloads beyond normal HTTPS transport;
- operator-grade RF calculation. "Nearest tower" means nearest mapped site by geographic distance, not necessarily the serving/available tower.

The dashboard uses 5-second HTTP polling because Streamlit is not acting as a push-message server. In v19, the timer is attached to the incident panel itself. It polls while the Streamlit session is active; it is not a background dispatch service. A production system would normally use a durable event/queue layer and authenticated responder clients.

The resident clock uses the device's current time, formatted explicitly in Myanmar time, and refreshes when returning to the page after sleep/backgrounding. Browsers may throttle timers in the background. The clock does not change incident timestamps, verify delivery or imply responder acknowledgement. See [v19 release notes](RELEASE_NOTES_V19.md) and [remaining risks](PROJECT_REVIEW_V19.md).
