"""Fail-closed distinction between source checks, usable inputs and background.

This policy does not calibrate, rescale or retrain any model. Unknown outputs
remain null, including in exports, instead of being classified as low exposure.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _state(status, label, reason, allowed=False):
    return {"status": status, "label": label, "reason": reason, "score_allowed": allowed}


def _time(value):
    if not isinstance(value, str) or not value:
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(stamp) else stamp


def _count(value):
    if isinstance(value, (bool, str)) or value is None:
        return None
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def assessment_state(kind, info, requested_mode, *, now=None):
    """Evaluate actual provenance, never infer an empty result from missing keys."""
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    unknown = _state("not_assessed", "Not assessed", "Usable live evidence is missing or incomplete. This is not a zero-risk result.")
    if kind == "compound":
        children = info.get("submodels", {})
        if not isinstance(children, Mapping):
            return unknown
        states = {name: assessment_state(name, children.get(name, {}), requested_mode, now=current)
                  for name in ("flood", "earthquake", "cyclone")}
        blocked = [name.title() for name, state in states.items() if not state["score_allowed"]]
        if blocked:
            state = _state("not_assessed", "Compound not assessed", "No combined score or population-impact estimate: " + ", ".join(blocked) + " has no usable event assessment. Missing/no-event components are not replaced by zero or historical scores. Review each hazard separately.")
        elif all(state["status"] == "live_assessed" for state in states.values()):
            state = _state("live_assessed", "Live-input planning assessment", "All three components have usable source inputs. Scores remain proxy-trained planning estimates, not confirmed damage.", True)
        else:
            state = _state("historical", "Historical / mixed planning scenario", "This combined result includes historical/demo inputs. It is not a live disaster prediction.", requested_mode != "gee")
        state["components"] = states
        return state
    actual_mode = info.get("mode")
    if actual_mode == "local":
        if requested_mode == "gee":
            return unknown
        return _state("historical", "Historical planning scenario", "Packaged historical/demo inputs are used. Current conditions are not established.", True)
    if actual_mode not in {"live", "gee"} or info.get("ok") is False:
        return unknown
    evidence = info.get("source_evidence", {})
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if kind == "flood":
        counts = evidence.get("hourly_image_counts", {})
        counts = counts if isinstance(counts, Mapping) else {}
        product_time = _time(evidence.get("product_time"))
        rows = _count(evidence.get("sample_rows"))
        requested = _count(evidence.get("requested_towers"))
        complete = all(_count(counts.get(hours, counts.get(str(hours)))) == hours for hours in (24, 72, 720))
        if (not complete or _count(evidence.get("unique_hourly_images")) != 720
                or not rows or rows != requested or product_time is None
                or _time(evidence.get("retrieved_at")) is None):
            return unknown
        age = (current - product_time).total_seconds() / 3600
        if age < -1 or age > 73:
            return _state("not_assessed", "Rainfall not assessed", "Rainfall evidence is outside the permitted 73-hour freshness window. Rerun the analysis; no current score or impact total is available.")
        return _state("live_assessed", "Observed-rainfall planning assessment", "Complete rainfall inputs passed the freshness checks. This model screens exposure using 30-day rainfall and site history; it does not detect or confirm flooding.", True)
    if kind == "earthquake":
        count = _count(evidence.get("accepted_events", info.get("event_count")))
        start, end = _time(evidence.get("query_start")), _time(evidence.get("query_end"))
        retrieved = _time(evidence.get("retrieved_at"))
        records = _count(evidence.get("retrieved_records"))
        ignored = _count(evidence.get("ignored_records"))
        if (count is None or records is None or ignored is None or records != count + ignored
                or start is None or end is None or retrieved is None or start >= end
                or evidence.get("possible_truncation") or records >= 2000):
            return unknown
        if count == 0:
            return _state("no_events", "No matching earthquakes detected", "The checked USGS query returned no qualifying earthquakes. No event-impact score or affected-population estimate is calculated. This does not establish safety or predict future earthquakes.")
        events = evidence.get("events", [])
        if not isinstance(events, list) or len(events) != count:
            return unknown
        for event in events:
            if not isinstance(event, Mapping):
                return unknown
            event_time = _time(event.get("time"))
            if event_time is None or not start <= event_time <= end:
                return unknown
        return _state("live_assessed", "Catalog-event planning assessment", "Qualifying catalog events were returned for the displayed query period. This is post-event exposure screening, not a future-earthquake prediction or evidence of ongoing damage.", True)
    if kind == "cyclone":
        storms, points = _count(info.get("active_storm_count")), _count(info.get("track_points"))
        products = _count(info.get("products_retrieved"))
        checked = _time(info.get("source_checked_at"))
        if checked is None or storms is None or points is None or products is None:
            return unknown
        if info.get("data_status") == "no_active_storm" and storms == points == products == 0:
            index_time = _time(info.get("index_updated_at"))
            if index_time is None:
                return unknown
            age = (current - index_time).total_seconds() / 3600
            if not -6 <= age <= float(info.get("index_max_age_hours", 72)):
                return unknown
            return _state("no_events", "No qualifying active cyclone forecast", "The checked source index returned no qualifying active forecast in the configured basins. No current-event score or affected-population estimate is calculated; this is not a guarantee of safety.")
        advisory = _time(info.get("storm_time"))
        if (info.get("data_status") != "active" or not storms or not points or not products or advisory is None):
            return unknown
        age = (current - advisory).total_seconds() / 3600
        if not -6 <= age <= float(info.get("advisory_max_age_hours", 36)):
            return _state("not_assessed", "Cyclone not assessed", "The forecast advisory is outside its freshness window. Rerun the analysis.")
        return _state("live_assessed", "Forecast-informed planning assessment", "Fresh qualifying forecast points were returned. Historical/site features still influence scores; they are not confirmed cyclone damage.", True)
    return unknown


def unassessed_frame(frame, kind, state):
    """Whitelist identity/context only; never leak legacy scores or actions."""
    identity = ["tower_id", "lat", "lon", "adm3_name", "analysis_area", "networks", "radios",
                "hazard_data_source", "hazard_data_timestamp"]
    out = frame[[name for name in identity if name in frame]].copy()
    out["hazard_type"] = kind
    out["assessment_status"] = state["status"]
    out["assessment_reason"] = state["reason"]
    for name in ("hazard_ai_score", "hazard_ai_pct", f"{kind}_ai_score", "population_affected",
                 "population_rerouted", "coverage_impact", "baseline_population_served"):
        out[name] = np.nan
    out["risk_level"] = "Not assessed"
    out["hazard_class"] = "Not assessed"
    out["assumed_unavailable"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["impact_status"] = "not_assessed"
    return out
