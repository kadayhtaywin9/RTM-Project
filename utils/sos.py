from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape


def sos_received_time(value: Any) -> str:
    """Show the service receipt clock, never replace it with browser/fetch time."""
    if not isinstance(value, str) or not value.strip():
        return "Receipt time unavailable"
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp) or stamp.tzinfo is None:
        return "Receipt time unavailable"
    return stamp.tz_convert("Asia/Yangon").strftime("%d %b %Y · %H:%M:%S MMT")


def fetch_sos_incidents(api_url: str, api_key: str = "", limit: int = 50, timeout: float = 3.0) -> tuple[list[dict[str, Any]], str | None]:
    if not api_url:
        return [], "SOS API URL is not configured."
    url = f"{api_url.rstrip('/')}/api/sos?limit={int(limit)}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        incidents = payload.get("incidents", []) if isinstance(payload, dict) else []
        return incidents if isinstance(incidents, list) else [], None
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return [], "SOS API rejected the dashboard key."
        return [], f"SOS API returned HTTP {exc.code}."
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return [], f"SOS API unavailable: {exc}"


def update_sos_status(api_url: str, incident_id: str, status: str, api_key: str = "", timeout: float = 3.0) -> tuple[bool, str | None]:
    url = f"{api_url.rstrip('/')}/api/sos/{incident_id}/status"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    data = json.dumps({"status": status}).encode("utf-8")
    req = urllib.request.Request(url, headers=headers, data=data, method="PATCH")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()
        return True, None
    except urllib.error.HTTPError as exc:
        return False, f"SOS API returned HTTP {exc.code}."
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, f"SOS API unavailable: {exc}"


def resolve_township(latitude: float, longitude: float, admin3_geo: dict) -> str | None:
    point = Point(float(longitude), float(latitude))
    for feature in admin3_geo.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            polygon = shape(geometry)
        except Exception:  # noqa: BLE001, S112 - preserve lookup behavior for malformed optional boundaries
            continue
        if polygon.covers(point):
            return feature.get("properties", {}).get("adm3_name")
    return None


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = np.radians(lat2.astype(float))
    dphi = p2 - p1
    dlambda = np.radians(lon2.astype(float) - lon1)
    a = np.sin(dphi / 2.0) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def nearest_tower(latitude: float, longitude: float, towers: pd.DataFrame) -> dict[str, Any] | None:
    if towers is None or towers.empty:
        return None
    usable = towers.dropna(subset=["lat", "lon"]).copy()
    if usable.empty:
        return None
    distances = _haversine_km(float(latitude), float(longitude), usable["lat"].to_numpy(), usable["lon"].to_numpy())
    idx = int(np.argmin(distances))
    row = usable.iloc[idx]
    result = row.to_dict()
    result["distance_km"] = float(distances[idx])
    return result


def enrich_sos_incident(incident: dict[str, Any], admin3_geo: dict, towers: pd.DataFrame) -> dict[str, Any]:
    enriched = dict(incident)
    lat = float(incident["latitude"])
    lon = float(incident["longitude"])
    enriched["township"] = resolve_township(lat, lon, admin3_geo) or "Outside mapped Yangon townships"
    tower = nearest_tower(lat, lon, towers)
    if tower:
        enriched["nearest_tower_id"] = tower.get("yangon_tower_id", "—")
        enriched["nearest_tower_lat"] = float(tower.get("lat"))
        enriched["nearest_tower_lon"] = float(tower.get("lon"))
        enriched["nearest_tower_km"] = float(tower.get("distance_km"))
        enriched["nearest_tower_networks"] = tower.get("networks", "—")
        enriched["nearest_tower_radios"] = tower.get("radios", "—")
        enriched["nearest_tower_township"] = tower.get("adm3_name", "—")
    else:
        enriched["nearest_tower_id"] = "—"
        enriched["nearest_tower_lat"] = None
        enriched["nearest_tower_lon"] = None
        enriched["nearest_tower_km"] = None
        enriched["nearest_tower_networks"] = "—"
        enriched["nearest_tower_radios"] = "—"
        enriched["nearest_tower_township"] = "—"
    return enriched
