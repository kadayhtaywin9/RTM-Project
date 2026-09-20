# RTM Project — Runtime

Dashboard and SOS runtime only. Training scripts, tests, unused datasets and historical documents are excluded.

Use 64-bit Python 3.11 or 3.12. On Windows, run `run_windows.bat` to set up and start the dashboard.

Start SOS in a second terminal from this folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
```

If using your existing `.venv-host`, substitute that environment in the command. Dashboard: http://localhost:8501 · SOS: http://localhost:8000

Configuration:
- `SOS_API_URL`: dashboard endpoint; default `http://127.0.0.1:8000`.
- `SOS_API_KEY`: set the same key in both processes before public deployment.
- `SOS_DB_PATH`: incident database; default `sos_service/sos.db`.
- `GEE_PROJECT_ID`: Earth Engine project for live rainfall; authenticate Earth Engine locally or configure `[gee]` credentials in Streamlit secrets.

Keep both processes running. External access needs hosting or a tunnel; phone GPS normally requires HTTPS. Demo hazard data needs no Earth Engine credentials.
