"""Visible source counts; records, periods and sampled rows are distinct units."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.gee_connector import GSMAP_DELAY_WARNING_HOURS, GSMAP_MAX_AGE_HOURS
from utils.source_evidence import product_age_hours


def _count(value) -> str:
    return "Not recorded" if value is None else f"{int(value):,}"


def _time(value) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y %H:%M UTC") if value else "Not recorded"


def render_source_evidence(context: dict) -> None:
    for item in context["source_records"]:
        record = item["retrieval"]
        kind = record["hazard_type"]
        data = record["details"]
        with st.container(border=True):
            provider = {"flood": "GEE GSMaP", "earthquake": "USGS", "cyclone": "JTWC"}.get(kind, kind.title())
            st.markdown(f"**{kind.title()} · {provider}**")
            if not record["external_inputs_used"]:
                if item["evidence_status"] in {"demo", "fallback"}:
                    st.write("No live external records used · historical inputs.")
                else:
                    st.write("Live evidence unavailable · counts unknown.")
                continue
            if kind == "flood":
                a, b, c = st.columns(3)
                a.metric("Unique hourly rainfall images", _count(data.get("unique_hourly_images")))
                b.metric("Rainfall aggregation periods", _count(len(data["aggregation_windows_hours"]) if "aggregation_windows_hours" in data else None))
                c.metric("Sampled tower rows", _count(data.get("sample_rows")))
                st.caption("Rainfall windows: 24 h / 72 h / 30 d · sampling: 10 km")
                st.caption(f"Source image: {_time(data.get('product_time'))} · Retrieved: {_time(data.get('retrieved_at'))}")
                age = product_age_hours(data)
                if age is not None:
                    st.caption(f"Product age: {age:.1f} h / {GSMAP_MAX_AGE_HOURS:g} h limit")
                    if age > GSMAP_MAX_AGE_HOURS:
                        st.warning("Rainfall product expired (>73 h). Rerun analysis.")
                    elif age > GSMAP_DELAY_WARNING_HOURS:
                        st.warning("Delayed rainfall product (>48 h); within the 73 h limit.")
            elif kind == "earthquake":
                a, b, c = st.columns(3)
                a.metric("Catalog records returned", _count(data.get("retrieved_records")))
                b.metric("Earthquakes used", _count(data.get("accepted_events")))
                c.metric("Query window", f"{_count(data.get('window_days'))} days")
                st.caption(f"Query: {_time(data.get('query_start'))} to {_time(data.get('query_end'))} · Retrieved: {_time(data.get('retrieved_at'))}")
                st.caption(f"Min magnitude: {data.get('minimum_magnitude', 'Not recorded')} · Ignored: {_count(data.get('ignored_records'))} · Bounds: tower extent + 4.5° · No time decay")
                events = pd.DataFrame(data.get("events", []))
                if not events.empty:
                    st.caption(f"Magnitude: {events.magnitude.min():.1f}–{events.magnitude.max():.1f}")
                    with st.expander(f"Events ({len(events):,})"):
                        st.dataframe(events.rename(columns={"time": "Event time (UTC)", "magnitude": "Magnitude", "depth_km": "Depth (km)", "place": "Place", "event_id": "Event ID"}), hide_index=True, width="stretch")
                elif data.get("accepted_events") == 0:
                    st.info("No matching earthquakes · impact not assessed.")
                if data.get("possible_truncation"):
                    st.warning("2,000-record limit reached · incomplete catalog.")
            elif kind == "cyclone":
                a, b, c = st.columns(3)
                a.metric("Forecast files retrieved", _count(data.get("products_retrieved")))
                b.metric("Fresh active storms", _count(data.get("active_storm_count")))
                c.metric("Current / forecast points", _count(data.get("track_points")))
                if data.get("active_storm_count", 0) > 0:
                    st.caption(f"Horizon: {_count(data.get('forecast_max_hours'))} h · Max wind: {data.get('max_wind_knots', 'Not recorded')} kt")
                else:
                    st.write("Forecast / wind: not assessed")
                st.caption(f"Advisory interval: {_time(data.get('advisory_start'))} to {_time(data.get('advisory_end'))} · Checked: {_time(data.get('source_checked_at'))}")
                st.caption(f"Basins: {', '.join(data.get('basins', [])) or 'Not recorded'} · Advisory age limit: {data.get('advisory_max_age_hours', 'Not recorded')} h")
                if data.get("data_status") == "no_active_storm":
                    st.info("No active forecast · impact not assessed.")
