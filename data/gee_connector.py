"""Google Earth Engine access with separate static and dynamic cache policies."""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from utils.caching import cache_data, cache_resource

SRTM_IMAGE_ID = "USGS/SRTMGL1_003"
DYNAMIC_WORLD_COLLECTION_ID = "GOOGLE/DYNAMICWORLD/V1"
DYNAMIC_WORLD_LOOKBACK_DAYS = 180
DYNAMIC_WORLD_CACHE_TTL_SECONDS = 21_600
DYNAMIC_WORLD_MIN_CONFIDENCE = 0.55
TERRAIN_SAMPLE_SCALE_METERS = 90
DYNAMIC_WORLD_SAMPLE_SCALE_METERS = 10
GSMAP_COLLECTION_ID = "JAXA/GPM_L3/GSMaP/v6/operational"
GSMAP_BAND = "hourlyPrecipRateGC"
GSMAP_CACHE_TTL_SECONDS = 900
GSMAP_MAX_AGE_HOURS = 48.0
GSMAP_FUTURE_TOLERANCE_HOURS = 1.0
GSMAP_WINDOW_HOURS = (24, 72, 24 * 30)
DYNAMIC_WORLD_PROBABILITY_BANDS = (
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
)
TERRAIN_FEATURE_COLUMNS = ("tower_id", "elevation", "slope")
DYNAMIC_WORLD_FEATURE_COLUMNS = ("tower_id", "land_cover", "land_cover_confidence")
STATIC_FEATURE_COLUMNS = (
    "tower_id",
    "elevation",
    "slope",
    "land_cover",
    "land_cover_confidence",
)
RAINFALL_FEATURE_COLUMNS = (
    "tower_id",
    "rainfall_24h",
    "rainfall_72h",
    "rainfall_30d",
)


def _as_utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} is not a valid timestamp") from exc
    if pd.isna(timestamp):
        raise RuntimeError(f"{label} is not a valid timestamp")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _integer_tower_ids(values: Iterable[Any], *, label: str) -> pd.Series:
    try:
        raw = pd.Series(list(values), dtype="object")
    except TypeError as exc:
        raise RuntimeError(f"{label} must be an iterable of tower IDs") from exc
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{label} contains a missing or non-numeric tower ID")
    if numeric.mod(1).ne(0).any():
        raise RuntimeError(f"{label} contains a non-integer tower ID")
    try:
        normalized = numeric.astype("int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} contains a tower ID outside the supported integer range") from exc
    if normalized.duplicated().any():
        duplicates = normalized[normalized.duplicated(keep=False)].drop_duplicates().tolist()
        raise RuntimeError(f"{label} contains duplicate tower IDs: {duplicates[:5]}")
    return normalized


def validate_rainfall_samples(
    frame: pd.DataFrame,
    tower_ids: Iterable[Any],
    timestamp: Any,
    *,
    now: Any | None = None,
) -> pd.DataFrame:
    """Validate and order a complete, fresh GSMaP sample for requested towers.

    This function is intentionally outside the cached fetch function. Calling it
    on every connector return ensures an otherwise valid cached result cannot be
    presented as live after its source timestamp becomes stale.
    """
    expected_ids = _integer_tower_ids(tower_ids, label="Requested tower set")
    if expected_ids.empty:
        raise RuntimeError("GSMaP validation requires at least one requested tower")

    observed_at = _as_utc_timestamp(timestamp, label="GSMaP source timestamp")
    current_time = _as_utc_timestamp(pd.Timestamp.now(tz="UTC") if now is None else now, label="Current time")
    age_hours = (current_time - observed_at).total_seconds() / 3600.0
    if age_hours < -GSMAP_FUTURE_TOLERANCE_HOURS:
        raise RuntimeError("GSMaP source timestamp is unexpectedly in the future")
    if age_hours > GSMAP_MAX_AGE_HOURS:
        raise RuntimeError(
            f"GSMaP source is stale ({age_hours:.1f} hours old; maximum is {GSMAP_MAX_AGE_HOURS:.0f})"
        )

    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("GSMaP sampling did not return a tabular result")  # noqa: TRY004 - remote sampling failure
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].unique().tolist()
        raise RuntimeError(f"GSMaP sampling returned duplicate columns: {duplicates}")
    missing_columns = [column for column in RAINFALL_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"GSMaP sampling is missing required columns: {missing_columns}")
    if frame.empty:
        raise RuntimeError("GSMaP sampling returned no tower features")

    validated = frame.copy()
    validated["tower_id"] = _integer_tower_ids(validated["tower_id"], label="GSMaP sample").to_numpy()
    expected_set = set(expected_ids.tolist())
    observed_set = set(validated["tower_id"].tolist())
    missing_ids = sorted(expected_set - observed_set)
    unexpected_ids = sorted(observed_set - expected_set)
    if missing_ids or unexpected_ids:
        details: list[str] = []
        if missing_ids:
            details.append(f"missing tower IDs {missing_ids[:5]}")
        if unexpected_ids:
            details.append(f"unrequested tower IDs {unexpected_ids[:5]}")
        raise RuntimeError(f"GSMaP sampling has incomplete tower coverage ({'; '.join(details)})")

    for column in RAINFALL_FEATURE_COLUMNS[1:]:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        values = validated[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"GSMaP sampling contains missing or non-finite {column} values")
        if (values < 0).any():
            raise RuntimeError(f"GSMaP sampling contains negative {column} values")
    tolerance = 1e-6
    if (validated["rainfall_24h"] > validated["rainfall_72h"] + tolerance).any() or (
        validated["rainfall_72h"] > validated["rainfall_30d"] + tolerance
    ).any():
        raise RuntimeError("GSMaP cumulative rainfall windows are not monotonic")

    order = pd.DataFrame({"tower_id": expected_ids.to_numpy()})
    return order.merge(validated, on="tower_id", how="left", sort=False, validate="one_to_one")


def validate_gsmap_hourly_coverage(timestamps_ms: Any, latest_ms: Any) -> dict[int, int]:
    """Require exactly one source image for every hour in each rainfall window."""
    if not isinstance(timestamps_ms, (list, tuple)):
        raise RuntimeError("GSMaP did not return an hourly source-image inventory")  # noqa: TRY004 - remote sampling failure
    try:
        latest = pd.to_datetime(int(latest_ms), unit="ms", utc=True)
        timestamps = pd.to_datetime(pd.Series(timestamps_ms, dtype="object"), unit="ms", utc=True, errors="coerce")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("GSMaP returned an invalid source-image timestamp") from exc
    if pd.isna(latest) or timestamps.isna().any():
        raise RuntimeError("GSMaP returned an invalid source-image timestamp")

    inventory = pd.DatetimeIndex(timestamps)
    earliest = latest - pd.Timedelta(hours=max(GSMAP_WINDOW_HOURS) - 1)
    inventory = inventory[(inventory >= earliest) & (inventory <= latest)]
    if inventory.duplicated().any():
        raise RuntimeError("GSMaP returned duplicate hourly source images")
    available = set(inventory.asi8.tolist())
    coverage: dict[int, int] = {}
    for hours in GSMAP_WINDOW_HOURS:
        expected = pd.date_range(end=latest, periods=hours, freq="h")
        missing_count = sum(value not in available for value in expected.asi8.tolist())
        if missing_count:
            raise RuntimeError(
                f"GSMaP hourly imagery is incomplete for the {hours}-hour window "
                f"({missing_count} missing hour{'s' if missing_count != 1 else ''})"
            )
        coverage[hours] = hours
    return coverage


@cache_resource(max_entries=4)
def initialize_earth_engine(project_id: str | None = None, _service_account_json: str | dict[str, Any] | None = None) -> Any:
    """Initialize GEE for local OAuth or Streamlit service-account deployment."""
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("earthengine-api is not installed; install requirements.txt first") from exc

    project_id = project_id or os.getenv("GEE_PROJECT_ID") or os.getenv("EARTHENGINE_PROJECT")
    credentials_value = _service_account_json or os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if credentials_value:
        info = json.loads(credentials_value) if isinstance(credentials_value, str) else dict(credentials_value)
        email = info.get("client_email")
        if not email:
            raise RuntimeError("GEE service-account JSON is missing client_email")
        credentials = ee.ServiceAccountCredentials(email, key_data=json.dumps(info))
        ee.Initialize(credentials=credentials, project=project_id)
    else:
        ee.Initialize(project=project_id)
    return ee


def _records(towers: pd.DataFrame) -> tuple[tuple[int, float, float], ...]:
    required = {"tower_id", "lat", "lon"}
    missing = sorted(required.difference(towers.columns))
    if missing:
        raise ValueError(f"Tower frame is missing required columns: {missing}")
    if towers.columns.duplicated().any():
        raise ValueError("Tower frame contains duplicate columns")
    try:
        tower_ids = _integer_tower_ids(towers["tower_id"], label="Tower frame")
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    latitudes = pd.to_numeric(towers["lat"], errors="coerce").to_numpy(dtype=float)
    longitudes = pd.to_numeric(towers["lon"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("Tower frame contains missing or non-finite coordinates")
    if (np.abs(latitudes) > 90).any() or (np.abs(longitudes) > 180).any():
        raise ValueError("Tower frame contains coordinates outside valid latitude/longitude ranges")
    return tuple(
        (int(tower_id), round(float(latitude), 7), round(float(longitude), 7))
        for tower_id, latitude, longitude in zip(tower_ids, latitudes, longitudes, strict=True)
    )


def _features_to_frame(ee: Any, collection: Any) -> pd.DataFrame:
    try:
        frame = ee.data.computeFeatures({"expression": collection, "fileFormat": "PANDAS_DATAFRAME"})
        return pd.DataFrame(frame)
    except Exception:  # noqa: BLE001 - any optimized-compute failure should use the documented API fallback
        payload = collection.getInfo()
        return pd.DataFrame([item.get("properties", {}) for item in payload.get("features", [])])


def _point_collection(ee: Any, records: tuple[tuple[int, float, float], ...]) -> Any:
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"tower_id": tower_id})
        for tower_id, lat, lon in records
    ])


def _feature_schema(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return only connector-contract columns, adding missing sampled bands as nulls."""
    out = frame.copy()
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    out = out.loc[:, list(columns)]
    out["tower_id"] = pd.to_numeric(out["tower_id"], errors="coerce").astype("Int64")
    return out


def _merge_tower_features(
    records: tuple[tuple[int, float, float], ...],
    *frames: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sampled features without dropping masked pixels or changing tower order."""
    merged = pd.DataFrame({"tower_id": pd.array([record[0] for record in records], dtype="Int64")})
    for frame in frames:
        addition = frame.copy()
        addition["tower_id"] = pd.to_numeric(addition["tower_id"], errors="coerce").astype("Int64")
        merged = merged.merge(addition, on="tower_id", how="left", sort=False, validate="many_to_one")
    return merged


@cache_data(ttl=None, max_entries=8)
def _terrain_features(records: tuple[tuple[int, float, float], ...], project_id: str | None, _credentials: Any) -> pd.DataFrame:
    """Sample terrain bands, which do not need time-based cache invalidation."""
    if not records:
        return _feature_schema(pd.DataFrame(), TERRAIN_FEATURE_COLUMNS)
    ee = initialize_earth_engine(project_id, _credentials)
    dem = ee.Image(SRTM_IMAGE_ID).select("elevation").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    stack = ee.Image.cat([dem, slope])
    sampled = stack.sampleRegions(
        collection=_point_collection(ee, records),
        properties=["tower_id"],
        scale=TERRAIN_SAMPLE_SCALE_METERS,
        geometries=False,
        tileScale=4,
    )
    return _feature_schema(_features_to_frame(ee, sampled), TERRAIN_FEATURE_COLUMNS)


def _build_dynamic_world_mosaic(ee: Any, collection: Any) -> Any:
    """Select, per pixel, the recent observation with the strongest class score."""
    def add_confidence(image: Any) -> Any:
        confidence = (
            image.select(list(DYNAMIC_WORLD_PROBABILITY_BANDS))
            .reduce(ee.Reducer.max())
            .rename("land_cover_confidence")
        )
        return image.select("label").rename("land_cover").addBands(confidence)

    return (
        collection.map(add_confidence)
        .qualityMosaic("land_cover_confidence")
        .select(["land_cover", "land_cover_confidence"])
    )


@cache_data(ttl=DYNAMIC_WORLD_CACHE_TTL_SECONDS, max_entries=8)
def _dynamic_world_features(
    records: tuple[tuple[int, float, float], ...],
    project_id: str | None,
    _credentials: Any,
) -> pd.DataFrame:
    """Sample a recent, confidence-ranked Dynamic World land-cover mosaic."""
    if not records:
        return _feature_schema(pd.DataFrame(), DYNAMIC_WORLD_FEATURE_COLUMNS)
    ee = initialize_earth_engine(project_id, _credentials)
    points = _point_collection(ee, records)
    now = pd.Timestamp.now(tz="UTC")
    collection = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION_ID)
        .filterDate(
            (now - pd.Timedelta(DYNAMIC_WORLD_LOOKBACK_DAYS, unit="D")).strftime("%Y-%m-%d"),
            (now + pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d"),
        )
        .filterBounds(points.geometry())
    )
    latest_ms = collection.aggregate_max("system:time_start").getInfo()
    if latest_ms is None:
        raise RuntimeError("GEE returned no recent Dynamic World imagery for the tower region")
    mosaic = _build_dynamic_world_mosaic(ee, collection)
    sampled = mosaic.sampleRegions(
        collection=points,
        properties=["tower_id"],
        scale=DYNAMIC_WORLD_SAMPLE_SCALE_METERS,
        geometries=False,
        tileScale=4,
    )
    result = _feature_schema(_features_to_frame(ee, sampled), DYNAMIC_WORLD_FEATURE_COLUMNS)
    result["land_cover_confidence"] = pd.to_numeric(result["land_cover_confidence"], errors="coerce")
    low_confidence = result["land_cover_confidence"].lt(DYNAMIC_WORLD_MIN_CONFIDENCE)
    result.loc[low_confidence.fillna(True), "land_cover"] = pd.NA
    return result


def _complete_rainfall_sum(collection: Any, hours: int, name: str) -> Any:
    """Mask totals where even one hourly pixel observation is missing."""
    return collection.sum().updateMask(collection.count().eq(hours)).rename(name)


@cache_data(ttl=GSMAP_CACHE_TTL_SECONDS, max_entries=8)
def _rainfall_features(records: tuple[tuple[int, float, float], ...], project_id: str | None, _credentials: Any) -> tuple[pd.DataFrame, str]:
    if not records:
        return pd.DataFrame(columns=list(RAINFALL_FEATURE_COLUMNS)), ""
    ee = initialize_earth_engine(project_id, _credentials)
    rainfall = ee.ImageCollection(GSMAP_COLLECTION_ID).select(GSMAP_BAND)
    now = pd.Timestamp.now(tz="UTC")
    recent = rainfall.filterDate((now - pd.Timedelta(days=45)).strftime("%Y-%m-%d"), (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    latest_ms = recent.aggregate_max("system:time_start").getInfo()
    if latest_ms is None:
        raise RuntimeError("GEE returned no recent GSMaP imagery")
    end = ee.Date(latest_ms).advance(1, "hour")
    thirty_day = rainfall.filterDate(end.advance(-30, "day"), end)
    hourly_timestamps = thirty_day.aggregate_array("system:time_start").getInfo()
    validate_gsmap_hourly_coverage(hourly_timestamps, latest_ms)
    stack = ee.Image.cat([
        _complete_rainfall_sum(rainfall.filterDate(end.advance(-24, "hour"), end), 24, "rainfall_24h"),
        _complete_rainfall_sum(rainfall.filterDate(end.advance(-72, "hour"), end), 72, "rainfall_72h"),
        _complete_rainfall_sum(thirty_day, 720, "rainfall_30d"),
    ])
    sampled = stack.sampleRegions(
        collection=_point_collection(ee, records),
        properties=["tower_id"],
        scale=10000,
        geometries=False,
        tileScale=4,
    )
    timestamp = pd.to_datetime(int(latest_ms), unit="ms", utc=True).isoformat()
    return _features_to_frame(ee, sampled), timestamp


class EarthEngineConnector:
    """Production-facing GEE connector optimized for batched tower sampling."""

    def __init__(self, project_id: str | None = None, service_account_json: str | dict[str, Any] | None = None) -> None:
        self.project_id = project_id
        self.service_account_json = service_account_json

    def static_features(self, towers: pd.DataFrame) -> pd.DataFrame:
        records = _records(towers)
        terrain = _terrain_features(records, self.project_id, self.service_account_json)
        land_cover = _dynamic_world_features(records, self.project_id, self.service_account_json)
        merged = _merge_tower_features(records, terrain, land_cover)
        return merged.loc[:, list(STATIC_FEATURE_COLUMNS)].copy()

    def rainfall_features(self, towers: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        records = _records(towers)
        frame, timestamp = _rainfall_features(records, self.project_id, self.service_account_json)
        validated = validate_rainfall_samples(frame, (record[0] for record in records), timestamp)
        return validated, timestamp

    def environmental_features(self, towers: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        static = self.static_features(towers)
        rainfall, timestamp = self.rainfall_features(towers)
        static["tower_id"] = pd.to_numeric(static["tower_id"], errors="coerce").astype("Int64")
        rainfall["tower_id"] = pd.to_numeric(rainfall["tower_id"], errors="coerce").astype("Int64")
        return static.merge(rainfall, on="tower_id", how="left", sort=False, validate="one_to_one"), timestamp
