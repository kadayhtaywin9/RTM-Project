"""Human-readable provenance for planning results, without invented confidence."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

SCORE_INTERPRETATION = (
    "Proxy-trained exposure planning score (raw 0–1; displayed 0–100), not a disaster probability, "
    "predicted tower failure, or measured outage."
)
EVIDENCE_LABELS = {
    "live": "Live inputs · planning model",
    "mixed": "Mixed live and cached inputs",
    "demo": "Demo / historical inputs",
    "fallback": "Cached fallback · live inputs unavailable",
    "unknown": "Unverified input provenance",
}
_SUMMARIES = {
    "live": "Live external inputs were used alongside packaged historical/site features. Live data does not validate the model's accuracy or confirm a tower outage.",
    "mixed": "Some hazard models used live inputs and others used cached fallback. Read each source status before comparing or acting on these planning scores.",
    "demo": "This is a historical/demo planning scenario. It does not describe confirmed current hazard conditions or tower outages.",
    "fallback": "Live inputs were not available for this result. Cached historical features were used; current conditions are not established by this analysis.",
    "unknown": "The recorded metadata is insufficient to verify the input provenance. Do not treat this result as a current-conditions assessment.",
}
_SOURCE_LABELS = {
    "flood": {
        "live": "GEE GSMaP rainfall + SRTM terrain; Dynamic World context only",
        "local": "Packaged rainfall, terrain and flood-history features",
    },
    "earthquake": {
        "live": "USGS recent earthquakes + packaged historical exposure",
        "local": "Packaged historical earthquake exposure; no live event signal",
    },
    "cyclone": {
        "live": "JTWC operational products + packaged historical cyclone exposure",
        "local": "Packaged historical cyclone exposure; current storm status unknown",
    },
}


def _timestamp(value: Any) -> str | None:
    if value is None or isinstance(value, (list, tuple, dict)):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).isoformat()


def _unique_timestamp_values(values: Any) -> dict[tuple[type, str], Any]:
    """Deduplicate raw values before expensive scalar datetime parsing.

    Keep missing values as distinct entries so partial observation clocks remain
    visible. Type-aware text keys also tolerate malformed unhashable inputs.
    """
    return {(type(value), str(value)): value for value in values}


def _single_status(info: Mapping[str, Any], requested_mode: str) -> str:
    actual_mode = str(info.get("mode", "")).lower().strip()
    if actual_mode in {"gee", "live"} and info.get("ok") is not False:
        return "live"
    if actual_mode == "local":
        return "demo" if requested_mode == "local" else "fallback"
    return "unknown"


def _source_context(
    hazard_type: str, info: Mapping[str, Any], requested_mode: str, frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    status = _single_status(info, requested_mode)
    # The current adapters have a fixed provider contract. Do not export raw
    # connector text: URLs or exception messages can expose project/credential
    # details. Actual modes determine provenance; names never determine status.
    provider_mode = "live" if status == "live" else "local"
    source = _SOURCE_LABELS.get(hazard_type, {}).get(provider_mode, "Source not recorded")
    if status == "unknown":
        source = "Source not recorded"
    candidates = [info.get("data_timestamp"), info.get("strongest_time"), info.get("storm_time")]
    frame_values = (
        _unique_timestamp_values(frame["hazard_data_timestamp"])
        if frame is not None and "hazard_data_timestamp" in frame else {}
    )
    if isinstance(info.get("data_timestamps"), (list, tuple)):
        candidates.extend(info["data_timestamps"])
    unique_values = _unique_timestamp_values(candidates)
    unique_values.update(frame_values)
    parsed_values = {key: _timestamp(value) for key, value in unique_values.items()}
    timestamps = sorted({stamp for stamp in parsed_values.values() if stamp is not None})
    recorded_values = (
        _unique_timestamp_values(info["data_timestamps"])
        if isinstance(info.get("data_timestamps"), (list, tuple)) else frame_values
    )
    has_missing = any(parsed_values[key] is None for key in recorded_values)
    timestamp_status = "not_recorded" if not timestamps else ("partly_recorded" if has_missing else "recorded")
    return {
        "hazard_type": hazard_type,
        "evidence_status": status,
        "evidence_label": EVIDENCE_LABELS[status],
        "source": source,
        "data_timestamps": timestamps,
        "timestamp_label": "Recorded event/product time (not a fetch-time guarantee)",
        "timestamp_status": timestamp_status,
        "source_checked_at": _timestamp(info.get("source_checked_at")),
        "data_status": info.get("data_status") if info.get("data_status") in {"active", "no_active_storm", "background_only"} else "",
        "message": _SUMMARIES[status],
    }


def build_result_context(
    frame: pd.DataFrame,
    run_info: Mapping[str, Any],
    requested_mode: str,
    completed_at: Any | None = None,
) -> dict[str, Any]:
    """Build JSON-safe evidence metadata from actual modes, not source-name guesses.

    A compound result has separate source clocks. Its latest input timestamp must
    never be advertised as the shared freshness of all three input models.
    """
    requested_mode = str(requested_mode).lower().strip()
    if requested_mode not in {"auto", "gee", "local"}:
        raise ValueError("requested_mode must be auto, gee, or local")
    hazard_type = str(run_info.get("hazard_type", ""))
    if not hazard_type and "hazard_type" in frame and not frame.empty:
        hazard_type = str(frame["hazard_type"].iloc[0])
    submodels: dict[str, dict[str, Any]] = {}
    if hazard_type == "compound":
        inputs = run_info.get("submodels", {})
        inputs = inputs if isinstance(inputs, Mapping) else {}
        for kind in ("flood", "earthquake", "cyclone"):
            child = inputs.get(kind, {})
            child = child if isinstance(child, Mapping) else {}
            submodels[kind] = _source_context(kind, child, requested_mode)
        statuses = {child["evidence_status"] for child in submodels.values()}
        if "unknown" in statuses:
            status = "unknown"
        elif statuses == {"live"}:
            status = "live"
        elif statuses == {"demo"}:
            status = "demo"
        elif "live" in statuses:
            status = "mixed"
        else:
            status = "fallback"
        source_records = list(submodels.values())
    else:
        source_records = [_source_context(hazard_type, run_info, requested_mode, frame)]
        status = source_records[0]["evidence_status"]

    limitations = [
        SCORE_INTERPRETATION,
        "Population counts use the packaged 2020 population baseline, not a current population census or observed affected people.",
        "Coverage impact is a what-if service-radius and tower-unavailability scenario; it is not measured network performance.",
        "Confirm official alerts, site condition and engineering evidence before any operational response.",
    ]
    if status != "live":
        limitations.append(_SUMMARIES[status])
    if any(source["timestamp_status"] != "recorded" for source in source_records):
        limitations.append("One or more source event/product timestamps are missing or only partly recorded; analysis completion time is not the observation time.")
    if hazard_type == "compound":
        limitations.append("Compound inputs have separate observation times; the newest source does not establish freshness of the others.")
    cyclone_context = submodels.get("cyclone") if hazard_type == "compound" else (source_records[0] if hazard_type == "cyclone" else None)
    if cyclone_context is not None:
        limitations.append("Cyclone training does not yet include a full IBTrACS history or verified tower-outage labels; live GEE rainfall and land cover are not cyclone-model inputs.")
        if cyclone_context["evidence_status"] != "live":
            limitations.append("Current cyclone activity is unknown in local/demo/fallback data; a zero event signal does not mean there is no storm.")
        elif cyclone_context["data_status"] == "no_active_storm":
            limitations.append("The fetched JTWC products contained no fresh active storm. This is a source-scoped result, not a guarantee of no cyclone hazard.")
    if hazard_type in {"flood", "compound"}:
        limitations.append("Dynamic World land cover is contextual information, not an input to the packaged Flood model.")
    completion = _timestamp(completed_at)
    if completed_at is not None and completion is None:
        raise ValueError("completed_at must be a valid timestamp")
    return {
        "schema_version": 1,
        "hazard_type": hazard_type,
        "requested_mode": requested_mode,
        "evidence_status": status,
        "evidence_label": EVIDENCE_LABELS[status],
        "evidence_summary": _SUMMARIES[status],
        "completed_at": completion,
        "sources": [source["source"] for source in source_records],
        "source_timestamps": [
            {"hazard_type": source["hazard_type"], "label": source["timestamp_label"], "timestamp": stamp}
            for source in source_records for stamp in source["data_timestamps"]
        ],
        "source_records": source_records,
        "submodels": submodels,
        "score_interpretation": SCORE_INTERPRETATION,
        "score_scale": {"raw": [0, 1], "display": [0, 100]},
        "population_reference_year": 2020,
        "limitations": limitations,
    }


def annotate_result_exports(frame: pd.DataFrame, context: Mapping[str, Any]) -> pd.DataFrame:
    """Keep evidence and interpretation attached when a result leaves the UI."""
    out = frame.copy()
    out["analysis_source_mode"] = context["requested_mode"]
    out["evidence_status"] = context["evidence_status"]
    out["score_interpretation"] = context["score_interpretation"]
    out["hazard_score_scale"] = "0–1 (multiply by 100 for display)"
    out["analysis_completed_at"] = context.get("completed_at") or ""
    out["population_reference_year"] = context["population_reference_year"]
    for kind, child in context.get("submodels", {}).items():
        out[f"{kind}_evidence_status"] = child["evidence_status"]
        out[f"{kind}_data_source"] = child["source"]
        out[f"{kind}_data_timestamps"] = " | ".join(child["data_timestamps"])
    return out
