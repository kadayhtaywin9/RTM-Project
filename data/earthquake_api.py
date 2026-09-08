"""USGS event ingestion for post-event telecom impact assessment."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from utils.caching import cache_data

USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_EVENT_TYPE = "earthquake"


def validate_usgs_geojson(payload: object) -> dict:
    """Reject HTTP-success error documents and malformed USGS catalog payloads."""
    if not isinstance(payload, Mapping):
        raise RuntimeError("USGS response is not a GeoJSON object")  # noqa: TRY004 - invalid remote data, not a caller type error
    if payload.get("type") != "FeatureCollection":
        raise RuntimeError("USGS response is not a GeoJSON FeatureCollection")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("USGS GeoJSON response is missing catalog metadata")  # noqa: TRY004 - remote data contract
    status = metadata.get("status")
    if status is not None:
        try:
            valid_status = int(status) == 200
        except (TypeError, ValueError):
            valid_status = False
        if not valid_status:
            raise RuntimeError(f"USGS catalog reported an unsuccessful status: {status!r}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise RuntimeError("USGS GeoJSON response is missing the features array")  # noqa: TRY004 - remote data contract
    declared_count = metadata.get("count")
    if declared_count is not None:
        try:
            count_matches = int(declared_count) == len(features)
        except (TypeError, ValueError):
            count_matches = False
        if not count_matches:
            raise RuntimeError("USGS GeoJSON feature count does not match its metadata")
    return dict(payload)


@cache_data(ttl=300, max_entries=16)
def _fetch_geojson(bounds: tuple[float, float, float, float], days: int, min_magnitude: float) -> dict:
    if int(days) <= 0:
        raise ValueError("USGS query window must be at least one day")
    if not math.isfinite(float(min_magnitude)):
        raise ValueError("USGS minimum magnitude must be finite")
    min_lat, max_lat, min_lon, max_lon = bounds
    now = pd.Timestamp.now(tz="UTC")
    params = {
        "format": "geojson",
        "starttime": (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endtime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "minmagnitude": min_magnitude,
        "eventtype": USGS_EVENT_TYPE,
        "orderby": "time",
        "limit": 2000,
    }
    request = Request(f"{USGS_ENDPOINT}?{urlencode(params)}", headers={"User-Agent": "GeoVisionAI/2.0"})
    with urlopen(request, timeout=15) as response:
        try:
            payload = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("USGS returned an invalid JSON response") from exc
    return validate_usgs_geojson(payload)


def _haversine(lat: np.ndarray, lon: np.ndarray, event_lat: float, event_lon: float) -> np.ndarray:
    lat1 = np.radians(lat); lon1 = np.radians(lon)
    lat2 = math.radians(event_lat); lon2 = math.radians(event_lon)
    delta_lat = lat2 - lat1; delta_lon = lon2 - lon1
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


class EarthquakeClient:
    """Retrieve recent earthquakes; this does not and must not predict earthquakes."""

    def tower_features(self, towers: pd.DataFrame, days: int = 30, min_magnitude: float = 3.0) -> tuple[pd.DataFrame, dict]:
        required = {"tower_id", "lat", "lon"}
        missing = sorted(required.difference(towers.columns))
        if missing:
            raise ValueError(f"Tower frame is missing required columns: {missing}")
        if towers.empty:
            return pd.DataFrame(columns=["tower_id", "magnitude", "depth", "distance_from_epicenter", "event_intensity"]), {
                "event_count": 0,
                "window_days": int(days),
                "magnitude": None,
                "place": "",
                "time": "",
                "source": "USGS FDSN Earthquake Catalog",
            }
        if int(days) <= 0:
            raise ValueError("USGS query window must be at least one day")
        lat = towers["lat"].to_numpy(float); lon = towers["lon"].to_numpy(float)
        if not np.isfinite(lat).all() or not np.isfinite(lon).all():
            raise ValueError("Tower frame contains missing or non-finite coordinates")
        bounds = (max(-90.0, float(lat.min()) - 4.5), min(90.0, float(lat.max()) + 4.5), max(-180.0, float(lon.min()) - 4.5), min(180.0, float(lon.max()) + 4.5))
        payload = validate_usgs_geojson(_fetch_geojson(bounds, int(days), float(min_magnitude)))
        best = np.zeros(len(towers), dtype=float)
        best_mag = np.zeros(len(towers), dtype=float)
        best_depth = np.zeros(len(towers), dtype=float)
        best_distance = np.full(len(towers), np.inf, dtype=float)
        strongest: dict = {"magnitude": None, "place": "", "time": ""}
        accepted_event_count = 0
        ignored_event_count = 0
        for feature in payload["features"]:
            if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
                raise RuntimeError("USGS catalog contains a malformed event feature")
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates") or [] if isinstance(geometry, Mapping) else []
            props = feature.get("properties") or {}
            if not isinstance(props, Mapping) or not props.get("type"):
                raise RuntimeError("USGS catalog contains an event with missing properties or event type")
            if str(props["type"]).lower() != USGS_EVENT_TYPE:
                ignored_event_count += 1
                continue
            if (
                not isinstance(geometry, Mapping) or geometry.get("type") != "Point"
                or not isinstance(coords, (list, tuple)) or len(coords) < 3
                or props.get("mag") is None or coords[2] is None
            ):
                raise RuntimeError("USGS earthquake is missing point geometry, magnitude or depth")
            try:
                event_lon = float(coords[0])
                event_lat = float(coords[1])
                raw_depth = float(coords[2] or 0)
                magnitude = float(props["mag"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("USGS earthquake contains invalid numeric fields") from exc
            if (
                not all(math.isfinite(value) for value in (event_lon, event_lat, raw_depth, magnitude))
                or abs(event_lat) > 90
                or abs(event_lon) > 180
            ):
                raise RuntimeError("USGS earthquake contains non-finite or out-of-range fields")
            accepted_event_count += 1
            depth = max(raw_depth, 0.0)
            distance = _haversine(lat, lon, event_lat, event_lon)
            intensity = np.clip(1.35 * np.clip((magnitude - 3.0) / 4.0, 0, 1) * np.exp(-depth / 180.0) * np.exp(-distance / 260.0), 0, 1)
            replace = intensity > best
            best[replace] = intensity[replace]
            best_mag[replace] = magnitude
            best_depth[replace] = depth
            best_distance[replace] = distance[replace]
            if strongest["magnitude"] is None or magnitude > strongest["magnitude"]:
                event_time = props.get("time")
                try:
                    event_time_text = pd.to_datetime(event_time, unit="ms", utc=True).isoformat() if event_time is not None else ""
                except (TypeError, ValueError, OverflowError):
                    event_time_text = ""
                strongest = {
                    "magnitude": magnitude,
                    "place": str(props.get("place") or ""),
                    "time": event_time_text,
                }
        frame = pd.DataFrame({
            "tower_id": towers["tower_id"].to_numpy(),
            "magnitude": best_mag,
            "depth": best_depth,
            "distance_from_epicenter": best_distance,
            "event_intensity": best,
        })
        return frame, {
            "event_count": accepted_event_count,
            "ignored_event_count": ignored_event_count,
            "window_days": days,
            **strongest,
            "source": "USGS FDSN Earthquake Catalog",
        }
