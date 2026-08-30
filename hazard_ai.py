"""GeoVision Disaster Impact AI runtime (Model 2, multi-hazard).

Hazard modes:
- flood: GEE GSMaP rainfall + SRTM terrain + project flood history
- earthquake: USGS recent event signal + project historical seismic exposure
- cyclone: GEE NOAA IBTrACS track/wind context + project historical cyclone exposure
- compound: learned meta-model over flood, earthquake and cyclone AI scores

All current outputs are hackathon/MVP *exposure/impact scores*, not calibrated
physical probabilities. See model cards/metadata in models/.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"

FLOOD_FEATURES = [
    "rain_30d_percentile",
    "flood_history_score",
    "elevation_risk",
    "slope_risk",
]

HAZARD_LABELS = {
    "flood": "Flood / Heavy Rain",
    "earthquake": "Earthquake",
    "cyclone": "Cyclone",
    "compound": "Compound",
}


def _load_xgb(path: Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model


@lru_cache(maxsize=1)
def _assets() -> dict[str, Any]:
    flood_model = _load_xgb(MODELS / "hazard_flood_xgb.json")
    flood_metadata = json.loads((MODELS / "hazard_model_metadata.json").read_text(encoding="utf-8"))
    multi_metadata = json.loads((MODELS / "multi_hazard_model_metadata.json").read_text(encoding="utf-8"))
    models = {
        "flood": flood_model,
        "earthquake": _load_xgb(MODELS / "hazard_earthquake_xgb.json"),
        "cyclone": _load_xgb(MODELS / "hazard_cyclone_xgb.json"),
        "compound": _load_xgb(MODELS / "hazard_compound_xgb.json"),
    }
    static = pd.read_csv(DATA / "tower_hazard_static_features.csv")
    rainfall = pd.read_csv(DATA / "yangon_rainfall_5y.csv")
    rainfall["date"] = pd.to_datetime(rainfall["date"])
    earthquakes = pd.read_csv(DATA / "earthquakes_near_yangon.csv")
    cyclones = pd.read_csv(DATA / "cyclone_labels.csv")
    return {
        "models": models,
        "flood_metadata": flood_metadata,
        "multi_metadata": multi_metadata,
        "static": static,
        "rainfall": rainfall,
        "earthquakes": earthquakes,
        "cyclones": cyclones,
    }


def hazard_model_status(hazard_type: str = "flood") -> dict[str, Any]:
    hazard_type = str(hazard_type).lower().strip()
    a = _assets()
    if hazard_type == "flood":
        m = a["flood_metadata"]
        limitations = [
            x for x in m.get("limitations", [])
            if "covers flood/heavy-rain exposure only" not in str(x).lower()
        ]
        return {
            "hazard_type": "flood",
            "label": HAZARD_LABELS["flood"],
            "model_name": m.get("model_name", "GeoVision Flood/Heavy-Rain AI"),
            "model_type": m.get("model_type", "XGBRegressor"),
            "training_rows": int(m.get("training_rows", 0)),
            "training_towers": int(m.get("training_towers", 0)),
            "training_snapshots": int(m.get("training_snapshots", 0)),
            "features": list(m.get("features_in_order", FLOOD_FEATURES)),
            "high_threshold": float(m.get("high_exposure_threshold", 0.50)),
            "very_high_threshold": float(m.get("very_high_exposure_threshold", 0.70)),
            "limitations": limitations,
            "cv_summary": dict(m.get("cv_summary", {})),
        }

    if hazard_type not in {"earthquake", "cyclone", "compound"}:
        raise ValueError(f"Unknown hazard_type: {hazard_type}")
    meta = a["multi_metadata"]
    m = meta["hazards"][hazard_type]
    th = meta.get("common_thresholds", {"high": 0.5, "very_high": 0.7})
    return {
        "hazard_type": hazard_type,
        "label": HAZARD_LABELS[hazard_type],
        "model_name": m.get("name", f"GeoVision {HAZARD_LABELS[hazard_type]} AI"),
        "model_type": "XGBRegressor",
        "training_rows": int(m.get("training_rows", 0)),
        "training_towers": int(m.get("training_towers", 0)),
        "training_snapshots": len(m.get("synthetic_event_intensity_levels", [])),
        "features": list(m.get("features_in_order", [])),
        "high_threshold": float(th.get("high", 0.50)),
        "very_high_threshold": float(th.get("very_high", 0.70)),
        "limitations": list(meta.get("limitations", [])),
        "cv_summary": dict(m.get("cv_summary", {})),
    }


def _radio_vulnerability(s: pd.Series) -> np.ndarray:
    text = s.fillna("").astype(str).str.upper()
    tech_count = text.str.count(",") + 1
    has_lte = text.str.contains("LTE").astype(float)
    has_umts = text.str.contains("UMTS").astype(float)
    value = 0.74 - 0.12 * (tech_count - 1) - 0.10 * has_lte - 0.03 * has_umts
    return np.clip(value.to_numpy(float), 0.25, 0.80)


def _prepare_common(towers: pd.DataFrame) -> pd.DataFrame:
    a = _assets()
    out = towers.copy()
    static = a["static"][["tower_id", "elevation_m", "slope_deg", "elevation_risk", "slope_risk"]].copy()
    out["tower_id"] = pd.to_numeric(out["tower_id"], errors="coerce").astype("Int64")
    static["tower_id"] = pd.to_numeric(static["tower_id"], errors="coerce").astype("Int64")
    # Avoid duplicate terrain fields if caller already carries them.
    for col in ["elevation_m", "slope_deg", "elevation_risk", "slope_risk"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out.merge(static, on="tower_id", how="left")
    for c in ["earthquake_score", "cyclone_score", "flood_history_score", "isolation_score", "elevation_risk", "slope_risk"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(0, 1)
    out["earthquake_history_score"] = out["earthquake_score"].clip(0, 1)
    out["cyclone_history_score"] = out["cyclone_score"].clip(0, 1)
    out["radio_vulnerability"] = _radio_vulnerability(out.get("radios", pd.Series("", index=out.index)))
    cell_count = pd.to_numeric(out.get("cell_count", pd.Series(1, index=out.index)), errors="coerce").fillna(1).clip(lower=1)
    max_cells = max(float(cell_count.max()), 1.0)
    out["site_redundancy_risk"] = 1.0 - np.log1p(cell_count) / np.log1p(max_cells)
    out["site_redundancy_risk"] = out["site_redundancy_risk"].clip(0, 1)
    return out


def _classify(out: pd.DataFrame, score: np.ndarray, source: str, data_timestamp: str | None, hazard_type: str) -> pd.DataFrame:
    score = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    out = out.copy()
    out["hazard_ai_score"] = score
    out["hazard_ai_pct"] = 100.0 * score
    out["hazard_class"] = pd.cut(
        score,
        bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)
    out["hazard_type"] = hazard_type
    out["hazard_label"] = HAZARD_LABELS[hazard_type]
    out["hazard_data_source"] = source
    out["hazard_data_timestamp"] = data_timestamp or ""
    return out


def _rain_percentile_by_area(current: pd.DataFrame, rainfall_history: pd.DataFrame) -> np.ndarray:
    history = rainfall_history.copy()
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"])
    lookup: dict[str, np.ndarray] = {}
    for pcode, sub in history.dropna(subset=["PCODE", "r1h"]).groupby("PCODE"):
        arr = np.sort(sub["r1h"].astype(float).to_numpy())
        if len(arr):
            lookup[str(pcode)] = arr
    out = np.full(len(current), 0.5, dtype=float)
    values = current["rain_30d_mm"].astype(float).to_numpy()
    pcodes = current["adm2_pcode"].astype(str).to_numpy()
    for i, (pcode, value) in enumerate(zip(pcodes, values)):
        arr = lookup.get(pcode)
        if arr is None or len(arr) == 0 or not np.isfinite(value):
            continue
        out[i] = float(np.searchsorted(arr, value, side="right") / len(arr))
    return np.clip(out, 0.0, 1.0)


def _predict_flood(df: pd.DataFrame, source: str, data_timestamp: str | None = None) -> pd.DataFrame:
    a = _assets()
    model: XGBRegressor = a["models"]["flood"]
    out = df.copy()
    for col in FLOOD_FEATURES:
        if col not in out.columns:
            raise ValueError(f"Flood feature missing: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(0, 1)
    score = model.predict(out[FLOOD_FEATURES].astype(float))
    out["flood_ai_score"] = np.clip(score, 0, 1)
    return _classify(out, out["flood_ai_score"], source, data_timestamp, "flood")


def initialize_gee(project_id: str | None = None, service_account_json: str | dict[str, Any] | None = None) -> Any:
    try:
        import ee
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("earthengine-api is not installed. Run pip install -r requirements.txt") from exc
    project_id = project_id or os.getenv("GEE_PROJECT_ID") or os.getenv("EARTHENGINE_PROJECT")
    service_account_json = service_account_json or os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        info = json.loads(service_account_json) if isinstance(service_account_json, str) else dict(service_account_json)
        email = info.get("client_email")
        if not email:
            raise RuntimeError("GEE service-account JSON is missing client_email.")
        credentials = ee.ServiceAccountCredentials(email, key_data=json.dumps(info))
        ee.Initialize(credentials=credentials, project=project_id)
    else:
        ee.Initialize(project=project_id)
    return ee


def _featurecollection_to_frame(ee: Any, fc: Any) -> pd.DataFrame:
    try:
        return ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    except Exception:
        info = fc.getInfo()
        return pd.DataFrame([f.get("properties", {}) for f in info.get("features", [])])


def local_flood_predictions(towers: pd.DataFrame, rain_date: str | None = None) -> pd.DataFrame:
    a = _assets()
    rain = a["rainfall"]
    base = _prepare_common(towers)
    dt = rain["date"].max() if rain_date is None else pd.Timestamp(rain_date)
    snap = rain[rain["date"] == dt][["PCODE", "r1h", "r1h_percentile"]].drop_duplicates("PCODE")
    base = base.merge(snap, left_on="adm2_pcode", right_on="PCODE", how="left")
    base["rain_30d_mm"] = pd.to_numeric(base["r1h"], errors="coerce").fillna(float(rain["r1h"].median()))
    base["rain_72h_mm"] = np.nan
    base["rain_30d_percentile"] = pd.to_numeric(base["r1h_percentile"], errors="coerce").fillna(0.5).clip(0, 1)
    base["flood_history_score"] = pd.to_numeric(base["flood_history_score"], errors="coerce").fillna(0).clip(0, 1)
    return _predict_flood(base, "Local cached rainfall/flood + cached SRTM", str(dt.date()))


def gee_flood_predictions(towers: pd.DataFrame, project_id: str | None = None, service_account_json: str | dict[str, Any] | None = None) -> pd.DataFrame:
    a = _assets()
    ee = initialize_gee(project_id=project_id, service_account_json=service_account_json)
    required = ["tower_id", "lat", "lon", "adm2_pcode", "flood_history_score"]
    missing = [c for c in required if c not in towers.columns]
    if missing:
        raise ValueError(f"Tower data missing required columns: {missing}")

    rain_all = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v6/operational").select("hourlyPrecipRateGC")
    now_utc = pd.Timestamp.now(tz="UTC")
    recent = rain_all.filterDate((now_utc - pd.Timedelta(days=45)).strftime("%Y-%m-%d"), (now_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    latest_ms = recent.aggregate_max("system:time_start").getInfo()
    if latest_ms is None:
        raise RuntimeError("GEE returned no recent GSMaP imagery for the last 45 days.")
    end = ee.Date(latest_ms).advance(1, "hour")
    rain30 = rain_all.filterDate(end.advance(-30, "day"), end).sum().rename("rain_30d_mm")
    rain72 = rain_all.filterDate(end.advance(-72, "hour"), end).sum().rename("rain_72h_mm")
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation_m")
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    stack = ee.Image.cat([rain30, rain72, dem, slope])

    features = []
    for row in towers[required].itertuples(index=False):
        features.append(ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"tower_id": int(row.tower_id), "adm2_pcode": str(row.adm2_pcode)}))
    sampled = stack.sampleRegions(collection=ee.FeatureCollection(features), properties=["tower_id", "adm2_pcode"], scale=90, geometries=False, tileScale=4)
    gee_df = _featurecollection_to_frame(ee, sampled)
    if gee_df.empty:
        raise RuntimeError("GEE sampling returned no tower features.")
    gee_df["tower_id"] = pd.to_numeric(gee_df["tower_id"], errors="coerce").astype("Int64")
    base = _prepare_common(towers)
    out = base.drop(columns=[c for c in ["elevation_m", "slope_deg"] if c in base.columns]).merge(
        gee_df[["tower_id", "rain_30d_mm", "rain_72h_mm", "elevation_m", "slope_deg"]], on="tower_id", how="left"
    )
    static = a["static"][["tower_id", "elevation_m", "slope_deg"]].rename(columns={"elevation_m": "elevation_m_cache", "slope_deg": "slope_deg_cache"})
    static["tower_id"] = pd.to_numeric(static["tower_id"], errors="coerce").astype("Int64")
    out = out.merge(static, on="tower_id", how="left")
    for c in ["rain_30d_mm", "rain_72h_mm", "elevation_m", "slope_deg"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["elevation_m"] = out["elevation_m"].fillna(out["elevation_m_cache"])
    out["slope_deg"] = out["slope_deg"].fillna(out["slope_deg_cache"])
    out.drop(columns=["elevation_m_cache", "slope_deg_cache"], inplace=True)
    out["rain_30d_percentile"] = _rain_percentile_by_area(out, a["rainfall"])
    out["elevation_risk"] = 1.0 - np.clip(out["elevation_m"].fillna(20.0) / 40.0, 0, 1)
    out["slope_risk"] = 1.0 - np.clip(out["slope_deg"].fillna(2.5) / 5.0, 0, 1)
    timestamp = pd.to_datetime(int(latest_ms), unit="ms", utc=True).isoformat()
    return _predict_flood(out, "Google Earth Engine — GSMaP + SRTM", timestamp)


def _haversine_km(lat: np.ndarray, lon: np.ndarray, ev_lat: float, ev_lon: float) -> np.ndarray:
    r = 6371.0088
    lat1 = np.radians(lat.astype(float)); lon1 = np.radians(lon.astype(float))
    lat2 = math.radians(float(ev_lat)); lon2 = math.radians(float(ev_lon))
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*math.cos(lat2)*np.sin(dlon/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _usgs_recent_earthquakes(towers: pd.DataFrame, days: int = 30) -> tuple[np.ndarray, dict[str, Any]]:
    now = pd.Timestamp.now(tz="UTC")
    lat_min = max(-90, float(towers["lat"].min()) - 4.5)
    lat_max = min(90, float(towers["lat"].max()) + 4.5)
    lon_min = max(-180, float(towers["lon"].min()) - 4.5)
    lon_max = min(180, float(towers["lon"].max()) + 4.5)
    params = {
        "format": "geojson",
        "starttime": (now - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        "endtime": (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "minlatitude": lat_min,
        "maxlatitude": lat_max,
        "minlongitude": lon_min,
        "maxlongitude": lon_max,
        "minmagnitude": 3.0,
        "orderby": "time",
        "limit": 2000,
    }
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "GeoVisionAI/1.0"})
    with urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    features = payload.get("features", [])
    lat = towers["lat"].to_numpy(float); lon = towers["lon"].to_numpy(float)
    signal = np.zeros(len(towers), dtype=float)
    strongest = {"mag": None, "place": "", "time": ""}
    for f in features:
        coords = (f.get("geometry") or {}).get("coordinates") or []
        props = f.get("properties") or {}
        if len(coords) < 3 or props.get("mag") is None:
            continue
        ev_lon, ev_lat, depth = float(coords[0]), float(coords[1]), max(float(coords[2] or 0), 0.0)
        mag = float(props["mag"])
        dist = _haversine_km(lat, lon, ev_lat, ev_lon)
        mag_norm = np.clip((mag - 3.0) / 4.0, 0, 1)
        depth_factor = np.exp(-depth / 180.0)
        event = np.clip(1.35 * mag_norm * depth_factor * np.exp(-dist / 260.0), 0, 1)
        signal = np.maximum(signal, event)
        if strongest["mag"] is None or mag > strongest["mag"]:
            ts = props.get("time")
            strongest = {
                "mag": mag,
                "place": str(props.get("place") or ""),
                "time": pd.to_datetime(ts, unit="ms", utc=True).isoformat() if ts else "",
            }
    return signal, {
        "event_count": int(len(features)),
        "strongest_magnitude": strongest["mag"],
        "strongest_place": strongest["place"],
        "strongest_time": strongest["time"],
        "window_days": days,
        "source": "USGS FDSN Earthquake Catalog",
    }


def _predict_earthquake(towers: pd.DataFrame, event_intensity: np.ndarray, source: str, timestamp: str | None = None) -> pd.DataFrame:
    a = _assets(); out = _prepare_common(towers)
    out["event_intensity"] = np.clip(np.asarray(event_intensity, dtype=float), 0, 1)
    features = a["multi_metadata"]["hazards"]["earthquake"]["features_in_order"]
    score = a["models"]["earthquake"].predict(out[features].astype(float))
    out["earthquake_ai_score"] = np.clip(score, 0, 1)
    return _classify(out, out["earthquake_ai_score"], source, timestamp, "earthquake")


def local_earthquake_predictions(towers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal = np.zeros(len(towers), dtype=float)
    result = _predict_earthquake(towers, signal, "Project historical earthquake exposure (no live event signal)", "")
    return result, {"event_count": 0, "source": "Local historical earthquake exposure", "message": "No live earthquake event signal used."}


def live_earthquake_predictions(towers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, meta = _usgs_recent_earthquakes(towers)
    timestamp = meta.get("strongest_time") or pd.Timestamp.now(tz="UTC").isoformat()
    result = _predict_earthquake(towers, signal, "USGS recent earthquakes + project historical exposure", timestamp)
    return result, meta


def _gee_cyclone_signal(towers: pd.DataFrame, project_id: str | None = None, service_account_json: str | dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    ee = initialize_gee(project_id=project_id, service_account_json=service_account_json)
    # IBTrACS is an archive/best-track dataset. We use the latest season available in GEE
    # near the selected region as observational context, not as a cyclone forecast.
    rect = ee.Geometry.Rectangle([
        float(towers["lon"].min()) - 1.0, float(towers["lat"].min()) - 1.0,
        float(towers["lon"].max()) + 1.0, float(towers["lat"].max()) + 1.0,
    ])
    fc = ee.FeatureCollection("NOAA/IBTrACS/v4").filter(ee.Filter.inList("BASIN", ["NI", "WP"])).filterBounds(rect.buffer(1800000))
    latest_season = fc.aggregate_max("SEASON").getInfo()
    if latest_season is None:
        raise RuntimeError("GEE IBTrACS returned no cyclone tracks near the analysis region.")
    latest = fc.filter(ee.Filter.eq("SEASON", latest_season))

    def add_xy(f: Any) -> Any:
        xy = f.geometry().coordinates()
        return f.set({"storm_lon": xy.get(0), "storm_lat": xy.get(1)})

    df = _featurecollection_to_frame(ee, latest.map(add_xy))
    if df.empty:
        raise RuntimeError("GEE IBTrACS latest-season query returned no track points.")
    lat = towers["lat"].to_numpy(float); lon = towers["lon"].to_numpy(float)
    signal = np.zeros(len(towers), dtype=float)
    max_wind = 0.0; best_name = ""; best_time = ""
    wind_cols = [c for c in ["USA_WIND", "WMO_WIND", "NEWDELHI_WIND", "TOKYO_WIND"] if c in df.columns]
    for _, row in df.iterrows():
        try:
            ev_lat = float(row.get("storm_lat")); ev_lon = float(row.get("storm_lon"))
        except Exception:
            continue
        winds = [pd.to_numeric(row.get(c), errors="coerce") for c in wind_cols]
        winds = [float(w) for w in winds if pd.notna(w) and float(w) > 0]
        wind = max(winds) if winds else 25.0
        dist = _haversine_km(lat, lon, ev_lat, ev_lon)
        wind_norm = np.clip((wind - 20.0) / 120.0, 0, 1)
        event = np.clip(1.25 * wind_norm * np.exp(-dist / 420.0), 0, 1)
        signal = np.maximum(signal, event)
        if wind > max_wind:
            max_wind = wind
            best_name = str(row.get("NAME") or "")
            best_time = str(row.get("ISO_TIME") or "")
    return signal, {
        "season": int(latest_season),
        "track_points": int(len(df)),
        "max_wind_knots": float(max_wind),
        "storm_name": best_name,
        "storm_time": best_time,
        "source": "Google Earth Engine NOAA IBTrACS v4",
        "archive_note": "IBTrACS is observational best-track/archive context, not a forecast feed.",
    }


def _local_cyclone_signal(towers: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    a = _assets(); c = a["cyclones"]
    signal = np.zeros(len(towers), dtype=float)
    if c.empty:
        return signal, {"track_points": 0, "source": "Local cyclone cache"}
    lat = towers["lat"].to_numpy(float); lon = towers["lon"].to_numpy(float)
    for _, row in c.iterrows():
        try:
            ev_lat = float(row["Latitude"]); ev_lon = float(row["Longitude"]); wind = float(row.get("WindSpeed", 25) or 25)
        except Exception:
            continue
        dist = _haversine_km(lat, lon, ev_lat, ev_lon)
        wind_norm = np.clip((wind - 20.0) / 120.0, 0, 1)
        signal = np.maximum(signal, np.clip(1.25 * wind_norm * np.exp(-dist / 420.0), 0, 1))
    return signal, {"track_points": int(len(c)), "source": "Local uploaded cyclone record", "archive_note": "Only the uploaded project cyclone record is used in local mode."}


def _predict_cyclone(towers: pd.DataFrame, event_intensity: np.ndarray, source: str, timestamp: str | None = None) -> pd.DataFrame:
    a = _assets(); out = _prepare_common(towers)
    out["event_intensity"] = np.clip(np.asarray(event_intensity, dtype=float), 0, 1)
    features = a["multi_metadata"]["hazards"]["cyclone"]["features_in_order"]
    score = a["models"]["cyclone"].predict(out[features].astype(float))
    out["cyclone_ai_score"] = np.clip(score, 0, 1)
    return _classify(out, out["cyclone_ai_score"], source, timestamp, "cyclone")


def local_cyclone_predictions(towers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, meta = _local_cyclone_signal(towers)
    result = _predict_cyclone(towers, signal, "Project cyclone exposure + local uploaded cyclone record", "")
    return result, meta


def gee_cyclone_predictions(towers: pd.DataFrame, project_id: str | None = None, service_account_json: str | dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, meta = _gee_cyclone_signal(towers, project_id=project_id, service_account_json=service_account_json)
    result = _predict_cyclone(towers, signal, "Google Earth Engine IBTrACS + project historical cyclone exposure", meta.get("storm_time") or str(meta.get("season", "")))
    return result, meta


def _predict_compound(towers: pd.DataFrame, flood: pd.DataFrame, earthquake: pd.DataFrame, cyclone: pd.DataFrame, source: str, timestamp: str | None = None) -> pd.DataFrame:
    a = _assets(); out = _prepare_common(towers)
    def score_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
        x = df[["tower_id", "hazard_ai_score"]].copy().rename(columns={"hazard_ai_score": col})
        x["tower_id"] = pd.to_numeric(x["tower_id"], errors="coerce").astype("Int64")
        return x
    out = out.merge(score_frame(flood, "flood_ai_score"), on="tower_id", how="left")
    out = out.merge(score_frame(earthquake, "earthquake_ai_score"), on="tower_id", how="left")
    out = out.merge(score_frame(cyclone, "cyclone_ai_score"), on="tower_id", how="left")
    for c in ["flood_ai_score", "earthquake_ai_score", "cyclone_ai_score", "isolation_score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(0, 1)
    features = a["multi_metadata"]["hazards"]["compound"]["features_in_order"]
    score = a["models"]["compound"].predict(out[features].astype(float))
    out["compound_ai_score"] = np.clip(score, 0, 1)
    return _classify(out, out["compound_ai_score"], source, timestamp, "compound")


def run_hazard_ai(
    towers: pd.DataFrame,
    mode: str = "auto",
    project_id: str | None = None,
    service_account_json: str | dict[str, Any] | None = None,
    rain_date: str | None = None,
    hazard_type: str = "flood",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a Model-2 hazard module.

    mode: auto | gee | local
      - gee means live external data path: GEE for flood/cyclone, USGS for earthquake.
      - auto tries the live path and falls back locally per module.
    hazard_type: flood | earthquake | cyclone | compound
    """
    mode = str(mode).lower().strip(); hazard_type = str(hazard_type).lower().strip()
    if mode not in {"auto", "gee", "local"}:
        raise ValueError("mode must be auto, gee, or local")
    if hazard_type not in HAZARD_LABELS:
        raise ValueError("hazard_type must be flood, earthquake, cyclone, or compound")

    if hazard_type == "flood":
        if mode in {"auto", "gee"}:
            try:
                result = gee_flood_predictions(towers, project_id=project_id, service_account_json=service_account_json)
                return result, {"mode": "gee", "ok": True, "hazard_type": hazard_type, "message": "Live GEE GSMaP + SRTM features used."}
            except Exception as exc:
                if mode == "gee":
                    raise
                result = local_flood_predictions(towers, rain_date=rain_date)
                return result, {"mode": "local", "ok": False, "hazard_type": hazard_type, "message": f"GEE flood data unavailable; local fallback used. {type(exc).__name__}: {exc}"}
        result = local_flood_predictions(towers, rain_date=rain_date)
        return result, {"mode": "local", "ok": True, "hazard_type": hazard_type, "message": "Local cached flood/heavy-rain features used."}

    if hazard_type == "earthquake":
        if mode in {"auto", "gee"}:
            try:
                result, meta = live_earthquake_predictions(towers)
                return result, {"mode": "live", "ok": True, "hazard_type": hazard_type, "message": "USGS recent earthquake events used with the trained Earthquake Impact AI.", **meta}
            except Exception as exc:
                if mode == "gee":
                    raise
                result, meta = local_earthquake_predictions(towers)
                return result, {"mode": "local", "ok": False, "hazard_type": hazard_type, "message": f"USGS earthquake data unavailable; local historical fallback used. {type(exc).__name__}: {exc}", **meta}
        result, meta = local_earthquake_predictions(towers)
        return result, {"mode": "local", "ok": True, "hazard_type": hazard_type, "message": "Local historical earthquake exposure used.", **meta}

    if hazard_type == "cyclone":
        if mode in {"auto", "gee"}:
            try:
                result, meta = gee_cyclone_predictions(towers, project_id=project_id, service_account_json=service_account_json)
                return result, {"mode": "gee", "ok": True, "hazard_type": hazard_type, "message": "GEE NOAA IBTrACS cyclone track/wind context used.", **meta}
            except Exception as exc:
                if mode == "gee":
                    raise
                result, meta = local_cyclone_predictions(towers)
                return result, {"mode": "local", "ok": False, "hazard_type": hazard_type, "message": f"GEE IBTrACS unavailable; local cyclone fallback used. {type(exc).__name__}: {exc}", **meta}
        result, meta = local_cyclone_predictions(towers)
        return result, {"mode": "local", "ok": True, "hazard_type": hazard_type, "message": "Local cyclone exposure used.", **meta}

    # Compound: run all three hazard modules using the same mode and feed their outputs to the meta-model.
    flood, frun = run_hazard_ai(towers, mode=mode, project_id=project_id, service_account_json=service_account_json, rain_date=rain_date, hazard_type="flood")
    quake, qrun = run_hazard_ai(towers, mode=mode, project_id=project_id, service_account_json=service_account_json, rain_date=rain_date, hazard_type="earthquake")
    cyc, crun = run_hazard_ai(towers, mode=mode, project_id=project_id, service_account_json=service_account_json, rain_date=rain_date, hazard_type="cyclone")
    sources = [str(flood.hazard_data_source.iloc[0]), str(quake.hazard_data_source.iloc[0]), str(cyc.hazard_data_source.iloc[0])]
    timestamps = [str(x) for x in [flood.hazard_data_timestamp.iloc[0], quake.hazard_data_timestamp.iloc[0], cyc.hazard_data_timestamp.iloc[0]] if str(x)]
    result = _predict_compound(towers, flood, quake, cyc, "Compound AI: " + " | ".join(sources), max(timestamps) if timestamps else "")
    ok = bool(frun.get("ok", True) and qrun.get("ok", True) and crun.get("ok", True))
    run_mode = "live" if all(r.get("mode") in {"gee", "live"} for r in [frun, qrun, crun]) else ("local" if all(r.get("mode") == "local" for r in [frun, qrun, crun]) else "mixed")
    return result, {
        "mode": run_mode,
        "ok": ok,
        "hazard_type": "compound",
        "message": "Compound AI combined Flood, Earthquake and Cyclone AI outputs.",
        "submodels": {"flood": frun, "earthquake": qrun, "cyclone": crun},
    }
