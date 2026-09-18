from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
DB_PATH = Path(os.getenv("SOS_DB_PATH", str(BASE / "sos.db")))
API_KEY = os.getenv("SOS_API_KEY", "").strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sos_incidents (
                id TEXT PRIMARY KEY,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                accuracy_m REAL,
                client_time TEXT,
                received_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'NEW',
                source TEXT NOT NULL DEFAULT 'resident-web',
                user_agent TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sos_received_at ON sos_incidents(received_at DESC)"
        )


class SOSCreate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0, le=100000)
    client_time: str | None = Field(default=None, max_length=64)


class StatusUpdate(BaseModel):
    status: Literal["NEW", "ACKNOWLEDGED", "RESOLVED"]


def require_dashboard_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid dashboard API key")


app = FastAPI(title="GeoVision SOS Service", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="resident-static")
allowed_origins = [x.strip() for x in os.getenv("SOS_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def resident_app() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "geovision-sos"}


@app.post("/api/sos", status_code=201)
def create_sos(payload: SOSCreate, request: Request) -> dict:
    incident_id = f"SOS-{uuid.uuid4().hex[:8].upper()}"
    received_at = utc_now()
    user_agent = request.headers.get("user-agent", "")[:300]
    with db() as conn:
        conn.execute(
            """
            INSERT INTO sos_incidents
                (id, latitude, longitude, accuracy_m, client_time, received_at, status, source, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, 'NEW', 'resident-web', ?)
            """,
            (
                incident_id,
                payload.latitude,
                payload.longitude,
                payload.accuracy_m,
                payload.client_time,
                received_at,
                user_agent,
            ),
        )
    return {
        "id": incident_id,
        "received_at": received_at,
        "status": "NEW",
        "message": "SOS received by GeoVision",
    }


@app.get("/api/sos")
def list_sos(
    limit: int = Query(default=50, ge=1, le=200),
    _: None = Depends(require_dashboard_key),
) -> dict:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, latitude, longitude, accuracy_m, client_time,
                   received_at, status, source
            FROM sos_incidents
            ORDER BY received_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"incidents": [dict(r) for r in rows]}


@app.patch("/api/sos/{incident_id}/status")
def update_status(
    incident_id: str,
    payload: StatusUpdate,
    _: None = Depends(require_dashboard_key),
) -> dict:
    with db() as conn:
        result = conn.execute(
            "UPDATE sos_incidents SET status = ? WHERE id = ?",
            (payload.status, incident_id),
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="SOS incident not found")
    return {"id": incident_id, "status": payload.status}
