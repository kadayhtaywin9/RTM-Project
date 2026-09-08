"""Keyless JTWC forecast ingestion and cyclone feature normalization.

JTWC does not expose a conventional authenticated JSON API. The U.S. Naval
Research Laboratory publishes the current-storm index and JTWC forecast aid as
ATCF ``.fst`` products, which this module discovers, validates and converts into
the model-facing tower schema.
"""
from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from utils.caching import cache_data

DEFAULT_JTWC_INDEX_URL = "https://science.nrlmry.navy.mil/atcf/index1.html"
DEFAULT_JTWC_FST_BASE_URL = "https://science.nrlmry.navy.mil/atcf/docs/current_storms/"
DEFAULT_JTWC_BASINS = ("IO", "WP")
DEFAULT_JTWC_INDEX_MAX_AGE_HOURS = 48.0
JTWC_SOURCE = "JTWC operational forecast via U.S. Naval Research Laboratory ATCF"
JTWC_TRACK_COLUMNS = [
    "storm_id",
    "storm_name",
    "advisory_time",
    "valid_time",
    "forecast_hour",
    "lat",
    "lon",
    "wind_speed",
    "pressure",
    "storm_type",
    "is_forecast",
    "source_url",
]

_FST_NAME = re.compile(r"(?i)^(?P<basin>[a-z]{2})(?P<number>\d{2})(?P<year>\d{4})\.fst$")
_FST_REFERENCE = re.compile(r"(?i)\b(?:io|wp|sh)\d{6}\.fst\b")
_WARNING_REFERENCE = re.compile(r"(?i)\b(?P<storm>(?:io|wp|sh)\d{6})\.wrn\b")
_INDEX_UPDATED_AT = re.compile(
    r"(?i)\blast\s+updated\s+(?P<timestamp>"
    r"(?:(?:mon|tue|wed|thu|fri|sat|sun)\s+)?"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+"
    r"\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+(?:utc|gmt)\s+\d{4})"
)


def cyclone_category(wind_knots: float) -> str:
    """Return a compact wind category for a one-minute sustained wind speed."""
    try:
        wind = float(wind_knots)
    except (TypeError, ValueError):
        wind = 0.0
    if not math.isfinite(wind) or wind < 34:
        return "Depression"
    if wind < 64:
        return "Tropical Storm"
    if wind < 83:
        return "Category 1"
    if wind < 96:
        return "Category 2"
    if wind < 113:
        return "Category 3"
    if wind < 137:
        return "Category 4"
    return "Category 5"


def _numeric_column(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    values = frame[name] if name in frame.columns else pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce")


def add_cyclone_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize optional forecast/archive fields into the model-facing schema."""
    out = frame.copy()
    out["wind_speed"] = _numeric_column(out, "wind_speed", 0.0).fillna(0.0)
    out["pressure"] = _numeric_column(out, "pressure", np.nan)
    out["distance_to_cyclone"] = _numeric_column(out, "distance_to_cyclone", np.inf)
    out["cyclone_category"] = out["wind_speed"].map(cyclone_category)
    return out


def parse_jtwc_coordinate(value: str, *, latitude: bool) -> float:
    """Parse an ATCF tenths-of-a-degree coordinate such as ``167N``/``956E``."""
    token = str(value).strip().upper()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([NSEW])", token)
    if not match:
        raise ValueError(f"Invalid JTWC coordinate: {value!r}")
    magnitude_text, hemisphere = match.groups()
    if latitude and hemisphere not in {"N", "S"}:
        raise ValueError(f"Expected a latitude coordinate, received {value!r}")
    if not latitude and hemisphere not in {"E", "W"}:
        raise ValueError(f"Expected a longitude coordinate, received {value!r}")
    magnitude = float(magnitude_text)
    if "." not in magnitude_text:
        magnitude /= 10.0
    if hemisphere in {"S", "W"}:
        magnitude *= -1.0
    limit = 90.0 if latitude else 180.0
    if not 0.0 <= abs(magnitude) <= limit:
        raise ValueError(f"JTWC coordinate is outside its valid range: {value!r}")
    return magnitude


def _empty_track() -> pd.DataFrame:
    return pd.DataFrame(columns=JTWC_TRACK_COLUMNS)


def _storm_id_from_url(source_url: str) -> str | None:
    filename = PurePosixPath(urlparse(source_url).path).name.lower()
    match = _FST_NAME.fullmatch(filename)
    return filename[:-4].upper() if match else None


def _atcf_season_year(basin: str, advisory_time: pd.Timestamp) -> int:
    """Return the ATCF season year, including July-December SH rollover."""
    return advisory_time.year + 1 if basin == "SH" and advisory_time.month >= 7 else advisory_time.year


def parse_jtwc_index_updated_at(index_text: str) -> pd.Timestamp | None:
    """Parse NRL's embedded ``Last updated`` value when the page provides one."""
    match = _INDEX_UPDATED_AT.search(str(index_text))
    if not match:
        return None
    timestamp = pd.to_datetime(match.group("timestamp"), utc=True, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def parse_jtwc_fst(text: str, *, source_url: str = "", storm_id_hint: str | None = None) -> pd.DataFrame:
    """Parse and strictly validate a JTWC ATCF forecast file.

    Duplicate ATCF records for wind-radius quadrants are reduced to one row per
    forecast hour. Only the latest advisory cycle and the official ``JTWC`` aid
    are retained. Malformed records are ignored; an entirely invalid product
    returns an empty frame so the client can distinguish it from stale data.
    """
    hint = (storm_id_hint or _storm_id_from_url(source_url) or "").upper().removesuffix(".FST")
    records: list[dict[str, Any]] = []
    for raw_line in str(text).splitlines():
        fields = [part.strip() for part in raw_line.split(",")]
        if len(fields) < 11:
            continue
        basin, number, cycle, aid = fields[0].upper(), fields[1], fields[2], fields[4].upper()
        if aid != "JTWC" or not re.fullmatch(r"[A-Z]{2}", basin) or not re.fullmatch(r"\d{1,2}", number):
            continue
        advisory_time = pd.to_datetime(cycle, format="%Y%m%d%H", utc=True, errors="coerce")
        if pd.isna(advisory_time):
            continue
        season_year = _atcf_season_year(basin, advisory_time)
        storm_id = f"{basin}{int(number):02d}{season_year:04d}"
        if hint and storm_id != hint:
            continue
        try:
            forecast_hour = int(fields[5])
            latitude = parse_jtwc_coordinate(fields[6], latitude=True)
            longitude = parse_jtwc_coordinate(fields[7], latitude=False)
            wind_speed = float(fields[8])
        except (TypeError, ValueError):
            continue
        if not 0 <= forecast_hour <= 168 or not 10 <= wind_speed <= 250:
            continue
        pressure_value = pd.to_numeric(fields[9], errors="coerce")
        pressure = float(pressure_value) if pd.notna(pressure_value) and 800 <= float(pressure_value) <= 1100 else np.nan
        storm_name = fields[27].strip().upper() if len(fields) > 27 else ""
        if storm_name in {"", "NONAME", "UNKNOWN"}:
            storm_name = storm_id
        records.append({
            "storm_id": storm_id,
            "storm_name": storm_name,
            "advisory_time": advisory_time,
            "valid_time": advisory_time + pd.to_timedelta(int(forecast_hour), unit="h"),
            "forecast_hour": forecast_hour,
            "lat": latitude,
            "lon": longitude,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "storm_type": fields[10].upper(),
            "is_forecast": forecast_hour > 0,
            "source_url": source_url,
        })
    if not records:
        return _empty_track()
    frame = pd.DataFrame.from_records(records, columns=JTWC_TRACK_COLUMNS)
    latest_cycle = frame.groupby("storm_id")["advisory_time"].transform("max")
    frame = frame[frame["advisory_time"] == latest_cycle]
    return (
        frame.sort_values(["storm_id", "forecast_hour", "pressure"], na_position="last")
        .drop_duplicates(["storm_id", "advisory_time", "forecast_hour"], keep="first")
        .reset_index(drop=True)
    )


def discover_jtwc_fst_urls(
    index_text: str,
    base_url: str,
    *,
    years: Iterable[int],
    basins: Iterable[str] = DEFAULT_JTWC_BASINS,
    max_files_per_basin: int = 12,
) -> list[str]:
    """Discover bounded current-season forecast URLs from an ATCF index page.

    NRL advertises active storms with ``.wrn`` links while the model consumes
    the matching ``.fst`` forecast product. Direct ``.fst`` references remain
    supported for private mirrors and backward-compatible configuration.
    """
    allowed_years = {int(year) for year in years}
    allowed_basins = {str(basin).upper() for basin in basins}
    names = {match.group(0).lower() for match in _FST_REFERENCE.finditer(str(index_text))}
    names.update(f"{match.group('storm').lower()}.fst" for match in _WARNING_REFERENCE.finditer(str(index_text)))
    selected: list[tuple[str, int, int, str]] = []
    for name in names:
        match = _FST_NAME.fullmatch(name)
        if not match:
            continue
        basin = match.group("basin").upper()
        year = int(match.group("year"))
        number = int(match.group("number"))
        if basin in allowed_basins and year in allowed_years:
            selected.append((basin, year, number, name))
    urls: list[str] = []
    normalized_base = base_url.rstrip("/") + "/"
    for basin in sorted(allowed_basins):
        candidates = sorted((item for item in selected if item[0] == basin), key=lambda item: (item[1], item[2]), reverse=True)
        urls.extend(urljoin(normalized_base, item[3]) for item in candidates[: max(1, int(max_files_per_basin))])
    return urls


def _validate_atcf_index(index_text: str) -> None:
    """Reject HTTP-200 error pages without misreporting them as no storms."""
    normalized = str(index_text).strip().lower()
    rejection_markers = ("request rejected", "requested url was rejected", "access denied")
    if not normalized or any(marker in normalized for marker in rejection_markers):
        raise RuntimeError("JTWC source returned a rejected or empty ATCF index page")
    has_product_reference = bool(_FST_REFERENCE.search(normalized) or _WARNING_REFERENCE.search(normalized))
    nrl_signature = (
        "automated tropical cyclone forecasting system" in normalized
        and "current global tropical cyclone activity" in normalized
    )
    if not has_product_reference and not nrl_signature:
        raise RuntimeError("JTWC source returned an unrecognized ATCF index page")


@cache_data(ttl=900, max_entries=64)
def _fetch_text(url: str, timeout_seconds: float = 15.0) -> str:
    request = Request(url, headers={"User-Agent": "GeoVisionAI/2.1 (+keyless-JTWC-ingest)", "Accept": "text/plain,text/html;q=0.9,*/*;q=0.1"})
    try:
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise RuntimeError("JTWC source response exceeded the 2 MB safety limit")
            charset = response.headers.get_content_charset() if hasattr(response.headers, "get_content_charset") else None
    except Exception as exc:
        raise RuntimeError(f"JTWC source request failed ({type(exc).__name__})") from exc
    return payload.decode(charset or "utf-8", errors="replace")


def _as_utc_timestamp(value: Any | None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _haversine_km(lat: np.ndarray, lon: np.ndarray, event_lat: float, event_lon: float) -> np.ndarray:
    lat1 = np.radians(lat.astype(float)); lon1 = np.radians(lon.astype(float))
    lat2 = math.radians(float(event_lat)); lon2 = math.radians(float(event_lon))
    delta_lat = lat2 - lat1; delta_lon = lon2 - lon1
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


class JTWCClient:
    """Retrieve recent official JTWC forecast tracks without an API key."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        index_url: str | None = None,
        timeout_seconds: float = 15.0,
        max_age_hours: float = 36.0,
        max_index_age_hours: float = DEFAULT_JTWC_INDEX_MAX_AGE_HOURS,
        basins: Iterable[str] = DEFAULT_JTWC_BASINS,
        max_files_per_basin: int = 12,
        max_track_points: int = 60,
        storm_ids: Iterable[str] | None = None,
        fetch_text: Callable[[str], str] | None = None,
    ) -> None:
        configured_base_url = base_url or os.getenv("JTWC_FST_BASE_URL")
        configured_index_url = index_url or os.getenv("JTWC_INDEX_URL")
        self.base_url = (configured_base_url or DEFAULT_JTWC_FST_BASE_URL).rstrip("/") + "/"
        self.index_url = configured_index_url or (self.base_url if configured_base_url else DEFAULT_JTWC_INDEX_URL)
        self.timeout_seconds = float(timeout_seconds)
        self.max_age_hours = float(max_age_hours)
        self.max_index_age_hours = float(max_index_age_hours)
        if not math.isfinite(self.max_age_hours) or self.max_age_hours <= 0:
            raise ValueError("JTWC advisory maximum age must be a positive finite number")
        if not math.isfinite(self.max_index_age_hours) or self.max_index_age_hours <= 0:
            raise ValueError("JTWC index maximum age must be a positive finite number")
        self.basins = tuple(str(basin).upper() for basin in basins)
        self.max_files_per_basin = max(1, int(max_files_per_basin))
        self.max_track_points = max(1, int(max_track_points))
        env_storm_ids = os.getenv("JTWC_STORM_IDS", "") if storm_ids is None else ""
        ids = list(storm_ids) if storm_ids is not None else env_storm_ids.split(",")
        self.storm_ids = tuple(self._normalize_storm_id(value) for value in ids if str(value).strip())
        self._fetcher = fetch_text or (lambda url: _fetch_text(url, self.timeout_seconds))

    @staticmethod
    def _normalize_storm_id(value: str) -> str:
        token = PurePosixPath(str(value).strip()).name.upper().removesuffix(".FST")
        if not re.fullmatch(r"(?:IO|WP|SH)\d{6}", token):
            raise ValueError(f"Invalid JTWC storm id: {value!r}")
        return token

    @staticmethod
    def _official_nrl_index(url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme.lower() == "https"
            and parsed.netloc.lower() == "science.nrlmry.navy.mil"
            and parsed.path.rstrip("/").lower() == "/atcf/index1.html"
        )

    def _candidate_urls(self, now: pd.Timestamp) -> tuple[list[str], dict[str, Any]]:
        source_metadata: dict[str, Any] = {
            "source_checked_at": now.isoformat(),
            "advisory_max_age_hours": self.max_age_hours,
            "index_max_age_hours": self.max_index_age_hours,
            "index_updated_at": "",
            "index_age_hours": None,
        }
        if self.storm_ids:
            return [urljoin(self.base_url, f"{storm_id.lower()}.fst") for storm_id in self.storm_ids], source_metadata
        index_text = self._fetcher(self.index_url)
        _validate_atcf_index(index_text)
        index_updated_at = parse_jtwc_index_updated_at(index_text)
        if index_updated_at is not None:
            index_age_hours = (now - index_updated_at).total_seconds() / 3600.0
            source_metadata["index_updated_at"] = index_updated_at.isoformat()
            source_metadata["index_age_hours"] = index_age_hours
            if index_age_hours < -6.0:
                raise RuntimeError("JTWC source index update timestamp is unexpectedly in the future")
            if index_age_hours > self.max_index_age_hours:
                raise RuntimeError(
                    "JTWC source index is too stale to verify current cyclone activity "
                    f"({index_age_hours:.1f} hours old; maximum is {self.max_index_age_hours:.0f})"
                )
        years = {now.year - 1, now.year}
        if "SH" in self.basins:
            years.add(now.year + 1)
        urls = discover_jtwc_fst_urls(
            index_text,
            self.base_url,
            years=years,
            basins=self.basins,
            max_files_per_basin=self.max_files_per_basin,
        )
        if not urls and index_updated_at is None and self._official_nrl_index(self.index_url):
            raise RuntimeError("JTWC source provided no storms and no verifiable index update timestamp")
        return urls, source_metadata

    def active_track(self, *, now: Any | None = None) -> pd.DataFrame:
        """Return non-stale current/forecast points, or an empty frame when none exist."""
        current_time = _as_utc_timestamp(now)
        urls, source_metadata = self._candidate_urls(current_time)
        if not urls:
            empty = _empty_track()
            empty.attrs["source_metadata"] = source_metadata
            return empty
        active: list[pd.DataFrame] = []
        fetched_count = 0
        valid_product_count = 0
        failed_fetch_count = 0
        invalid_product_count = 0
        for url in urls:
            try:
                product = self._fetcher(url)
            except Exception:  # noqa: BLE001 - one unavailable storm must not block the other advertised products
                failed_fetch_count += 1
                continue
            fetched_count += 1
            frame = parse_jtwc_fst(product, source_url=url)
            if frame.empty:
                invalid_product_count += 1
                continue
            valid_product_count += 1
            advisory_time = frame["advisory_time"].max()
            age_hours = (current_time - advisory_time).total_seconds() / 3600.0
            if -6.0 <= age_hours <= self.max_age_hours:
                active.append(frame)
        if failed_fetch_count or invalid_product_count:
            raise RuntimeError(
                "JTWC source returned incomplete current-storm data "
                f"({failed_fetch_count} failed and {invalid_product_count} invalid products)"
            )
        if not active:
            if fetched_count == 0:
                raise RuntimeError("JTWC source advertised forecast files but none could be retrieved")
            if valid_product_count == 0:
                raise RuntimeError("JTWC source returned no valid official forecast records")
            raise RuntimeError("JTWC source advertised only stale or future-dated forecast products; current storm status is unknown")
        track = pd.concat(active, ignore_index=True)
        latest_cycle = track.groupby("storm_id")["advisory_time"].transform("max")
        result = (
            track[track["advisory_time"] == latest_cycle]
            .sort_values(["storm_id", "forecast_hour"])
            .drop_duplicates(["storm_id", "forecast_hour"], keep="first")
            .reset_index(drop=True)
        )
        result.attrs["source_metadata"] = source_metadata
        return result

    def _forecast_track_metadata(self, track: pd.DataFrame) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for row in track.sort_values(["storm_id", "forecast_hour"]).head(self.max_track_points).itertuples(index=False):
            pressure = None if pd.isna(row.pressure) else float(row.pressure)
            serialized.append({
                "storm_id": str(row.storm_id),
                "storm_name": str(row.storm_name),
                "forecast_hour": int(row.forecast_hour),
                "valid_time": _as_utc_timestamp(row.valid_time).isoformat(),
                "lat": float(row.lat),
                "lon": float(row.lon),
                "wind_speed": float(row.wind_speed),
                "pressure": pressure,
            })
        return serialized

    @staticmethod
    def _empty_context(towers: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "tower_id": towers["tower_id"].to_numpy(),
            "distance_to_cyclone": np.full(len(towers), np.inf),
            "wind_speed": np.zeros(len(towers)),
            "pressure": np.full(len(towers), np.nan),
            "cyclone_category": ["Not applicable"] * len(towers),
            "storm_id": [""] * len(towers),
            "storm_name": [""] * len(towers),
            "forecast_hour": np.full(len(towers), np.nan),
            "forecast_valid_time": [""] * len(towers),
            "track_status": ["No active JTWC cyclone"] * len(towers),
        })

    def tower_features(self, towers: pd.DataFrame, *, now: Any | None = None) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
        """Derive bounded tower impact signals from active JTWC forecast tracks."""
        required = {"tower_id", "lat", "lon"}
        missing = sorted(required.difference(towers.columns))
        if missing:
            raise ValueError(f"Tower frame is missing required columns: {missing}")
        track = self.active_track(now=now)
        context = self._empty_context(towers)
        base_meta: dict[str, Any] = {
            "source": JTWC_SOURCE,
            "source_url": self.index_url if not self.storm_ids else self.base_url,
            "active_storm_count": int(track["storm_id"].nunique()) if not track.empty else 0,
            "track_points": len(track),
            "forecast_track": self._forecast_track_metadata(track),
            **dict(track.attrs.get("source_metadata", {})),
        }
        if track.empty:
            return np.zeros(len(towers), dtype=float), {
                **base_meta,
                "data_status": "no_active_storm",
                "max_wind_knots": 0.0,
                "storm_id": "",
                "storm_name": "",
                "storm_time": "",
            }, context

        lat = towers["lat"].to_numpy(float); lon = towers["lon"].to_numpy(float)
        signal = np.zeros(len(towers), dtype=float)
        nearest_distance = np.full(len(towers), np.inf, dtype=float)
        selected_wind = np.zeros(len(towers), dtype=float)
        selected_pressure = np.full(len(towers), np.nan, dtype=float)
        selected_hour = np.full(len(towers), np.nan, dtype=float)
        selected_id = np.full(len(towers), "", dtype=object)
        selected_name = np.full(len(towers), "", dtype=object)
        selected_time = np.full(len(towers), "", dtype=object)
        for row in track.itertuples(index=False):
            distance = _haversine_km(lat, lon, float(row.lat), float(row.lon))
            wind_norm = np.clip((float(row.wind_speed) - 20.0) / 120.0, 0.0, 1.0)
            lead_confidence = math.exp(-max(float(row.forecast_hour), 0.0) / 168.0)
            event = np.clip(1.25 * wind_norm * np.exp(-distance / 420.0) * lead_confidence, 0.0, 1.0)
            replace = (event > signal) | ((event == signal) & (distance < nearest_distance))
            signal = np.maximum(signal, event)
            nearest_distance[replace] = distance[replace]
            selected_wind[replace] = float(row.wind_speed)
            selected_pressure[replace] = float(row.pressure) if pd.notna(row.pressure) else np.nan
            selected_hour[replace] = float(row.forecast_hour)
            selected_id[replace] = str(row.storm_id)
            selected_name[replace] = str(row.storm_name)
            selected_time[replace] = _as_utc_timestamp(row.valid_time).isoformat()
        context = pd.DataFrame({
            "tower_id": towers["tower_id"].to_numpy(),
            "distance_to_cyclone": nearest_distance,
            "wind_speed": selected_wind,
            "pressure": selected_pressure,
            "cyclone_category": [cyclone_category(value) for value in selected_wind],
            "storm_id": selected_id,
            "storm_name": selected_name,
            "forecast_hour": selected_hour,
            "forecast_valid_time": selected_time,
            "track_status": ["Active JTWC forecast"] * len(towers),
        })
        primary = track.sort_values(["wind_speed", "forecast_hour"], ascending=[False, True]).iloc[0]
        latest_advisory = _as_utc_timestamp(track["advisory_time"].max()).isoformat()
        return signal, {
            **base_meta,
            "data_status": "active",
            "max_wind_knots": float(track["wind_speed"].max()),
            "storm_id": str(primary["storm_id"]),
            "storm_name": str(primary["storm_name"]),
            "storm_time": latest_advisory,
        }, context
