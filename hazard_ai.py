"""GeoVision Disaster Impact AI runtime (Model 2, multi-hazard).

Hazard modes:
- flood: GEE GSMaP rainfall + SRTM terrain + project flood history
- earthquake: USGS recent event signal + project historical seismic exposure
- cyclone: keyless JTWC current/forecast track context + project historical cyclone exposure
- compound: learned meta-model over flood, earthquake and cyclone AI scores

All current outputs are hackathon/MVP *exposure/impact scores*, not calibrated
physical probabilities. See model cards/metadata in models/.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data.cyclone_api import JTWCClient
from data.earthquake_api import EarthquakeClient
from data.gee_connector import (
    EarthEngineConnector,
    initialize_earth_engine,
    validate_rainfall_samples,
)

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
    normalization = a["multi_metadata"].get("feature_normalization", {})
    max_cells = float(normalization.get("cell_count_max", 32.0))
    if not np.isfinite(max_cells) or max_cells < 1.0:
        raise ValueError("Invalid fixed cell_count_max in multi-hazard model metadata")
    cell_count = (
        pd.to_numeric(out.get("cell_count", pd.Series(1.0, index=out.index)), errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0, upper=max_cells)
    )
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
        right=False,
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
    """Compatibility adapter for the cached connector layer."""
    return initialize_earth_engine(project_id, service_account_json)


def _featurecollection_to_frame(ee: Any, fc: Any) -> pd.DataFrame:
    try:
        return ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    except Exception:  # noqa: BLE001 - Earth Engine raises several backend-specific exception types
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
    required = ["tower_id", "lat", "lon", "adm2_pcode", "flood_history_score"]
    missing = [c for c in required if c not in towers.columns]
    if missing:
        raise ValueError(f"Tower data missing required columns: {missing}")

    connector = EarthEngineConnector(project_id=project_id, service_account_json=service_account_json)
    gee_df, timestamp = connector.environmental_features(towers)
    if gee_df.empty:
        raise RuntimeError("GEE sampling returned no tower features.")
    gee_df = validate_rainfall_samples(gee_df, towers["tower_id"], timestamp)
    gee_df["tower_id"] = pd.to_numeric(gee_df["tower_id"], errors="coerce").astype("Int64")
    gee_df = gee_df.rename(columns={
        "rainfall_24h": "rain_24h_mm",
        "rainfall_72h": "rain_72h_mm",
        "rainfall_30d": "rain_30d_mm",
        "elevation": "elevation_m",
        "slope": "slope_deg",
    })
    base = _prepare_common(towers)
    out = base.drop(columns=[c for c in ["elevation_m", "slope_deg"] if c in base.columns]).merge(
        gee_df[[c for c in ["tower_id", "rain_24h_mm", "rain_72h_mm", "rain_30d_mm", "elevation_m", "slope_deg", "land_cover", "land_cover_confidence"] if c in gee_df]],
        on="tower_id",
        how="left",
    )
    static = a["static"][["tower_id", "elevation_m", "slope_deg"]].rename(columns={"elevation_m": "elevation_m_cache", "slope_deg": "slope_deg_cache"})
    static["tower_id"] = pd.to_numeric(static["tower_id"], errors="coerce").astype("Int64")
    out = out.merge(static, on="tower_id", how="left")
    for c in ["rain_24h_mm", "rain_30d_mm", "rain_72h_mm", "elevation_m", "slope_deg"]:
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["elevation_m"] = out["elevation_m"].fillna(out["elevation_m_cache"])
    out["slope_deg"] = out["slope_deg"].fillna(out["slope_deg_cache"])
    out.drop(columns=["elevation_m_cache", "slope_deg_cache"], inplace=True)
    out["rain_30d_percentile"] = _rain_percentile_by_area(out, a["rainfall"])
    out["elevation_risk"] = 1.0 - np.clip(out["elevation_m"].fillna(20.0) / 40.0, 0, 1)
    out["slope_risk"] = 1.0 - np.clip(out["slope_deg"].fillna(2.5) / 5.0, 0, 1)
    return _predict_flood(out, "GEE GSMaP + SRTM; Dynamic World shown as context only", timestamp)


def _recent_earthquake_features(towers: pd.DataFrame, days: int = 30) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, raw = EarthquakeClient().tower_features(towers, days=days)
    meta = {
        "event_count": int(raw.get("event_count", 0)),
        "strongest_magnitude": raw.get("magnitude"),
        "strongest_place": raw.get("place", ""),
        "strongest_time": raw.get("time", ""),
        "window_days": int(raw.get("window_days", days)),
        "source": raw.get("source", "USGS FDSN Earthquake Catalog"),
    }
    return features, meta


def _haversine_km(lat: np.ndarray, lon: np.ndarray, ev_lat: float, ev_lon: float) -> np.ndarray:
    r = 6371.0088
    lat1 = np.radians(lat.astype(float)); lon1 = np.radians(lon.astype(float))
    lat2 = math.radians(float(ev_lat)); lon2 = math.radians(float(ev_lon))
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*math.cos(lat2)*np.sin(dlon/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _usgs_recent_earthquakes(towers: pd.DataFrame, days: int = 30) -> tuple[np.ndarray, dict[str, Any]]:
    features, meta = _recent_earthquake_features(towers, days=days)
    return features["event_intensity"].to_numpy(float), meta


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
    event_features, meta = _recent_earthquake_features(towers)
    timestamp = meta.get("strongest_time") or pd.Timestamp.now(tz="UTC").isoformat()
    result = _predict_earthquake(towers, event_features["event_intensity"].to_numpy(float), "USGS recent earthquakes + project historical exposure", timestamp)
    result = result.merge(
        event_features[["tower_id", "magnitude", "depth", "distance_from_epicenter"]],
        on="tower_id",
        how="left",
    )
    return result, meta


def _jtwc_cyclone_signal(towers: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    """Build the live event signal from fresh JTWC operational forecast products."""
    return JTWCClient().tower_features(towers)


def _local_cyclone_signal(towers: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    """Return historical background without replaying an old storm as current."""
    signal = np.zeros(len(towers), dtype=float)
    context = pd.DataFrame({
        "tower_id": towers["tower_id"].to_numpy(),
        "distance_to_cyclone": np.nan,
        "wind_speed": np.nan,
        "pressure": np.nan,
        "cyclone_category": "Not assessed",
        "track_status": "Background only; current storm status unknown",
    })
    return signal, {
        "data_status": "background_only",
        "track_points": 0,
        "source": "Project historical cyclone exposure; no current event signal",
        "archive_rows": len(_assets()["cyclones"]),
        "archive_note": "The limited historical record informs background exposure only. Current cyclone activity is unknown in local mode.",
    }, context


def _predict_cyclone(towers: pd.DataFrame, event_intensity: np.ndarray, source: str, timestamp: str | None = None) -> pd.DataFrame:
    a = _assets(); out = _prepare_common(towers)
    out["event_intensity"] = np.clip(np.asarray(event_intensity, dtype=float), 0, 1)
    features = a["multi_metadata"]["hazards"]["cyclone"]["features_in_order"]
    score = a["models"]["cyclone"].predict(out[features].astype(float))
    out["cyclone_ai_score"] = np.clip(score, 0, 1)
    return _classify(out, out["cyclone_ai_score"], source, timestamp, "cyclone")


def local_cyclone_predictions(towers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, meta, context = _local_cyclone_signal(towers)
    result = _predict_cyclone(towers, signal, "Project historical cyclone exposure; current storm status unknown", "")
    result = result.merge(context, on="tower_id", how="left")
    result["coastal_exposure"] = np.clip(0.7 * result["elevation_risk"] + 0.3 * result["flood_history_score"], 0, 1)
    result["tower_vulnerability"] = result["radio_vulnerability"]
    return result, meta


def live_cyclone_predictions(towers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, meta, context = _jtwc_cyclone_signal(towers)
    active = meta.get("data_status") == "active"
    source = (
        "JTWC current/forecast track + project historical cyclone exposure"
        if active
        else "JTWC reports no fresh active storm; project background cyclone exposure"
    )
    result = _predict_cyclone(towers, signal, source, meta.get("storm_time") or "")
    result = result.merge(context, on="tower_id", how="left")
    result["coastal_exposure"] = np.clip(0.7 * result["elevation_risk"] + 0.3 * result["flood_history_score"], 0, 1)
    result["tower_vulnerability"] = result["radio_vulnerability"]
    return result, meta


def gee_cyclone_predictions(
    towers: pd.DataFrame,
    project_id: str | None = None,
    service_account_json: str | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible name; live cyclone data now comes from keyless JTWC."""
    del project_id, service_account_json
    return live_cyclone_predictions(towers)


def _compound_tower_ids(frame: pd.DataFrame, label: str) -> pd.Index:
    """Validate stable integer IDs before aligning independently produced outputs."""
    if "tower_id" not in frame or frame.columns.duplicated().any():
        raise ValueError(f"{label} must include one unambiguous tower_id column")
    numeric = pd.to_numeric(frame["tower_id"], errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if (
        frame["tower_id"].map(lambda value: isinstance(value, (bool, np.bool_))).any()
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values >= 2**63).any()
        or (values != np.floor(values)).any()
    ):
        raise ValueError(f"{label} tower IDs must be non-negative finite integers")
    ids = pd.Index(numeric.astype("int64"), name="tower_id")
    if ids.has_duplicates:
        raise ValueError(f"{label} contains duplicate tower IDs")
    return ids


def _compound_child_scores(frame: pd.DataFrame, requested: pd.Index, label: str) -> np.ndarray:
    ids = _compound_tower_ids(frame, label)
    if len(ids) != len(requested) or not ids.difference(requested).empty or not requested.difference(ids).empty:
        raise ValueError(f"{label} tower IDs must exactly match the requested towers")
    if "hazard_ai_score" not in frame:
        raise ValueError(f"{label} is missing hazard_ai_score")
    scores = pd.to_numeric(frame["hazard_ai_score"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{label} scores must be finite values between 0 and 1")
    return pd.Series(scores, index=ids).reindex(requested).to_numpy(float)


def _predict_compound(towers: pd.DataFrame, flood: pd.DataFrame, earthquake: pd.DataFrame, cyclone: pd.DataFrame, source: str, timestamp: str | None = None) -> pd.DataFrame:
    requested = _compound_tower_ids(towers, "Requested")
    if requested.empty:
        raise ValueError("Compound analysis requires at least one requested tower")
    # A missing upstream result is unknown, never a zero-exposure observation.
    child_scores = {
        f"{label}_ai_score": _compound_child_scores(frame, requested, label.capitalize())
        for label, frame in [("flood", flood), ("earthquake", earthquake), ("cyclone", cyclone)]
    }
    a = _assets(); out = _prepare_common(towers)
    prepared_ids = _compound_tower_ids(out, "Prepared")
    if len(prepared_ids) != len(requested) or not prepared_ids.difference(requested).empty:
        raise ValueError("Prepared tower IDs must exactly match the requested towers")
    out["tower_id"] = prepared_ids
    out = out.set_index("tower_id").loc[requested].reset_index()
    for column, scores in child_scores.items():
        out[column] = scores
    out["isolation_score"] = pd.to_numeric(out["isolation_score"], errors="coerce").fillna(0).clip(0, 1)
    features = a["multi_metadata"]["hazards"]["compound"]["features_in_order"]
    score = a["models"]["compound"].predict(out[features].astype(float))
    out["compound_ai_score"] = np.clip(score, 0, 1)
    return _classify(out, out["compound_ai_score"], source, timestamp, "compound")


def _run_hazard_ai(
    towers: pd.DataFrame,
    mode: str = "auto",
    project_id: str | None = None,
    service_account_json: str | dict[str, Any] | None = None,
    rain_date: str | None = None,
    hazard_type: str = "flood",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a Model-2 hazard module.

    mode: auto | gee | local
      - gee means strict live external data: GEE for flood, USGS for earthquake,
        and keyless JTWC public products for cyclone. The name is retained for API compatibility.
      - auto tries the live path and falls back locally per module.
    hazard_type: flood | earthquake | cyclone | compound
    """
    mode = str(mode).lower().strip(); hazard_type = str(hazard_type).lower().strip()
    if mode not in {"auto", "gee", "local"}:
        raise ValueError("mode must be auto, gee, or local")
    if hazard_type not in HAZARD_LABELS:
        raise ValueError("hazard_type must be flood, earthquake, cyclone, or compound")
    if hazard_type == "compound" and _compound_tower_ids(towers, "Requested").empty:
        raise ValueError("Compound analysis requires at least one requested tower")

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
                result, meta = live_cyclone_predictions(towers)
                message = (
                    "Fresh JTWC operational current/forecast track used."
                    if meta.get("data_status") == "active"
                    else "No fresh active JTWC cyclone found; the result shows background exposure only."
                )
                return result, {"mode": "live", "ok": True, "hazard_type": hazard_type, "message": message, **meta}
            except Exception as exc:
                if mode == "gee":
                    raise
                result, meta = local_cyclone_predictions(towers)
                return result, {"mode": "local", "ok": False, "hazard_type": hazard_type, "message": f"JTWC operational products unavailable; local cyclone fallback used. {type(exc).__name__}: {exc}", **meta}
        result, meta = local_cyclone_predictions(towers)
        return result, {"mode": "local", "ok": True, "hazard_type": hazard_type, "message": "Historical cyclone background used; current storm status is unknown.", **meta}

    # Compound: run all three hazard modules using the same mode and feed their outputs to the meta-model.
    flood, frun = run_hazard_ai(towers, mode=mode, project_id=project_id, service_account_json=service_account_json, rain_date=rain_date, hazard_type="flood")
    quake, qrun = run_hazard_ai(towers, mode=mode, project_id=project_id, service_account_json=service_account_json, rain_date=rain_date, hazard_type="earthquake")
    cyc, crun = run_hazard_ai(towers, mode=mode, project_id=project_id, service_account_json=service_account_json, rain_date=rain_date, hazard_type="cyclone")
    sources = [str(info.get("source", "")) for info in [frun, qrun, crun]]
    timestamps = [str(info["data_timestamp"]) for info in [frun, qrun, crun] if info.get("data_timestamp")]
    result = _predict_compound(towers, flood, quake, cyc, "Compound AI: " + " | ".join(sources), max(timestamps) if timestamps else "")
    ok = bool(frun.get("ok", True) and qrun.get("ok", True) and crun.get("ok", True))
    run_mode = "live" if all(r.get("mode") in {"gee", "live"} for r in [frun, qrun, crun]) else ("local" if all(r.get("mode") == "local" for r in [frun, qrun, crun]) else "mixed")
    return result, {
        "mode": run_mode,
        "ok": ok,
        "hazard_type": "compound",
        "message": "Compound AI combined Flood, Earthquake and Cyclone AI outputs.",
        "timestamp_scope": "latest_input_only",
        "submodels": {"flood": frun, "earthquake": qrun, "cyclone": crun},
    }


def run_hazard_ai(
    towers: pd.DataFrame,
    mode: str = "auto",
    project_id: str | None = None,
    service_account_json: str | dict[str, Any] | None = None,
    rain_date: str | None = None,
    hazard_type: str = "flood",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a hazard module and expose the output's observation provenance.

    Modes are auto (live with fallback), gee (strict live; legacy name), and
    local. Compound run_info includes each child's source and data_timestamp;
    its legacy aggregate timestamp describes only the latest dated input, not
    the freshness of all three inputs. Empty timestamps mean not available.
    """
    result, run_info = _run_hazard_ai(
        towers, mode=mode, project_id=project_id,
        service_account_json=service_account_json, rain_date=rain_date,
        hazard_type=hazard_type,
    )
    run_info = dict(run_info)
    for output_column, info_key in [
        ("hazard_data_source", "source"),
        ("hazard_data_timestamp", "data_timestamp"),
    ]:
        values = (
            result[output_column].fillna("").astype(str).unique().tolist()
            if output_column in result else []
        )
        if info_key == "source":
            source = " | ".join(value for value in values if value)
            if run_info.get("source") and run_info["source"] != source:
                run_info["provider_source"] = run_info["source"]
            run_info["source"] = source
        else:
            # Do not pick the first timestamp if an adapter ever emits mixed ages.
            run_info["data_timestamp"] = values[0] if len(values) == 1 else ""
            if len(values) > 1:
                run_info["data_timestamps"] = values
                run_info["timestamp_scope"] = "per_tower_output"
    return result, run_info
