"""Explicit units and observation clocks for external hazard inputs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def utc_stamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return "" if pd.isna(stamp) else stamp.isoformat()


def source_evidence_record(kind: str, info: Mapping) -> dict:
    """Whitelist provenance fields; never copy credentials or raw error text."""
    live = info.get("mode") in {"live", "gee"} and info.get("ok") is not False
    record = {"hazard_type": kind, "external_inputs_used": live, "details": {}}
    if not live:
        return record
    source = info.get("source_evidence", {})
    source = source if isinstance(source, Mapping) else {}
    if kind == "cyclone":
        fields = (
            "active_storm_count", "track_points", "forecast_max_hours", "products_requested",
            "products_retrieved", "products_used", "basins", "advisory_max_age_hours",
            "index_max_age_hours", "max_wind_knots", "data_status",
        )
        details = {key: info[key] for key in fields if key in info}
        for key in ("source_checked_at", "advisory_start", "advisory_end", "storm_time", "index_updated_at"):
            if info.get(key):
                details[key] = utc_stamp(info[key])
    else:
        fields = (
            ("aggregation_windows_hours", "hourly_image_counts", "unique_hourly_images",
             "sample_rows", "requested_towers", "sample_scale_m", "max_age_hours", "cache_ttl_seconds")
            if kind == "flood" else
            ("retrieved_records", "accepted_events", "ignored_records", "window_days",
             "minimum_magnitude", "query_bounds", "query_limit", "cache_ttl_seconds", "possible_truncation")
        )
        details = {key: source[key] for key in fields if key in source}
        for key in ("retrieved_at", "product_time", "window_start", "window_end_exclusive", "query_start", "query_end"):
            if source.get(key):
                details[key] = utc_stamp(source[key])
        if kind == "earthquake":
            details.setdefault("accepted_events", info.get("event_count"))
            details.setdefault("window_days", info.get("window_days"))
            details["events"] = [
                {key: event.get(key) for key in ("event_id", "time", "magnitude", "depth_km", "place")}
                for event in source.get("events", []) if isinstance(event, Mapping)
            ]
    record["details"] = details
    return record


def product_age_hours(details: Mapping, now: Any = None) -> float | None:
    stamp = utc_stamp(details.get("product_time"))
    if not stamp:
        return None
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    return (current - pd.Timestamp(stamp)).total_seconds() / 3600
