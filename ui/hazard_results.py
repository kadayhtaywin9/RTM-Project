"""Map-first hazard results with explicit evidence and scenario interpretation."""
from __future__ import annotations

import json
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.cesium_map import render_map
from engine.decision_engine import attach_coverage_scenario
from ui.source_evidence import render_source_evidence
from utils.live_assessment import unassessed_frame
from utils.result_context import annotate_result_exports, build_result_context
from utils.scenario_report import build_scenario_report, spreadsheet_safe_csv
from utils.visualization import EXPOSURE_COLORSCALE, RISK_COLORS, hazard_summary


def _evidence_strip(context: dict) -> None:
    status = context["evidence_status"]
    state = context["assessment"]
    label = {
        "live": "Live inputs", "demo": "Historical inputs", "fallback": "Historical fallback",
        "mixed": "Mixed inputs", "unknown": "Not assessed",
    }.get(status, "Not assessed")
    if not state["score_allowed"]:
        status = "unknown"
        label = "No matching events" if state["status"] == "no_events" else "Not assessed"
    providers = {"flood": "GEE", "earthquake": "USGS", "cyclone": "JTWC"}
    sources = " / ".join(providers.get(item["hazard_type"], item["hazard_type"].title())
                         for item in context["source_records"]
                         if item["retrieval"]["external_inputs_used"])
    completed = context.get("completed_at")
    completed_text = pd.Timestamp(completed).strftime("%d %b %Y · %H:%M UTC") if completed else "Time not recorded"
    st.markdown(
        f'<section class="gv-evidence gv-evidence-{escape(status)}" aria-label="Result evidence">'
        f'<span class="gv-evidence-label">{escape(label)}{(" · " + escape(sources)) if sources else ""}</span>'
        f'<small>Run {escape(completed_text)}</small></section>',
        unsafe_allow_html=True,
    )


def _source_details(context, status):
    with st.expander("Sources & model", expanded=False):
        render_source_evidence(context)
        st.caption("Proxy-trained exposure scores · uncalibrated · geographic coverage · population baseline: 2020.")
        if status.get("features"):
            st.caption("Model inputs: " + ", ".join(status["features"]))
        if status.get("training_rows") is not None:
            st.caption(f"{status['model_type']} · {status['training_towers']:,} towers · {status['training_rows']:,} proxy/synthetic training rows")
        st.caption("Scenario assumes surviving and unassessed towers are available; RF, capacity, power and backhaul are excluded. Full provenance and limitations are included in the JSON export.")


def _exposure_map(hz, hazard_type, selected_areas, run_info, base_map, add_track, scenario_enabled=False):
    fig = base_map(selected_areas, hz, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    if hazard_type == "cyclone":
        add_track(fig, run_info.get("forecast_track", []))
    customdata = np.column_stack([
        hz.tower_id, hz.adm3_name, hz.hazard_ai_score * 100, hz.risk_level,
        hz.coverage_impact, hz.assumed_unavailable.map({True: "Yes", False: "No"}),
    ])
    fig.add_trace(go.Scattermap(
        lat=hz.lat, lon=hz.lon, mode="markers",
        marker={"size": np.where(hz.assumed_unavailable.fillna(False), 8, 6), "opacity": 1.0,
                "color": hz.hazard_ai_score * 100, "cmin": 0, "cmax": 100,
                "colorscale": EXPOSURE_COLORSCALE, "showscale": True,
                "colorbar": {"title": {"text": "Exposure<br>0–100"}, "thickness": 10, "len": .65}},
        customdata=customdata,
        hovertemplate=("<b>Tower %{customdata[0]}</b><br>%{customdata[1]}"
                       "<br>Exposure score: %{customdata[2]:.1f} / 100 · %{customdata[3]}"
                       + ("<br>Assumed unavailable: %{customdata[5]}"
                          "<br>People losing coverage in scenario: %{customdata[4]:,.0f}" if scenario_enabled else "")
                       + "<extra>Planning score — not observed outages</extra>"),
        name="Tower exposure",
    ))
    fig.update_layout(height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False)
    return fig


def _unassessed_map(frame, selected_areas, base_map):
    fig = base_map(selected_areas, frame, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    fig.add_trace(go.Scattermap(
        lat=frame.lat, lon=frame.lon, mode="markers", name="Not assessed",
        marker={"size": 5, "color": "#94a3b8", "opacity": .7, "showscale": False},
        customdata=np.column_stack([frame.tower_id, frame.adm3_name]),
        hovertemplate="<b>Tower %{customdata[0]}</b><br>%{customdata[1]}<br>Not assessed — no event-impact estimate<extra>Unknown, not zero risk</extra>",
    ))
    fig.update_layout(height=480, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False)
    return fig


def _render_unassessed(result, context, hazard_type, selected_areas, base_map, map_config):
    state = context["assessment"]
    hz = unassessed_frame(result, hazard_type, state)
    st.info("No usable event assessment. Review source status below or switch to Demo data.")
    if "components" in state:
        st.dataframe(pd.DataFrame([
            {"Hazard": kind.title(), "Status": item["label"]}
            for kind, item in state["components"].items()
        ]), hide_index=True, width="stretch")
    st.markdown("#### Population impact")
    for column, label in zip(st.columns(3), ("Affected people", "Coverage loss", "Rerouted"), strict=True):
        column.metric(label, "Not assessed")
    st.markdown("#### Assessment map")
    render_map(_unassessed_map(hz, selected_areas, base_map), width="stretch", config=map_config, key=f"hazard_unassessed_map_{hazard_type}")
    st.caption("Gray: not assessed")
    exported = annotate_result_exports(hz, context)
    report = {
        "report_version": "GeoVision v21", "result_context": context,
        "assessed_tower_count": 0, "listed_tower_count": len(hz),
        "scenario": {"enabled": False, "status": "not_assessed", "reason": state["reason"]},
        "totals": {"population_directly_affected": None, "population_losing_coverage": None, "population_rerouted": None},
    }
    a, b = st.columns(2)
    a.download_button("Availability CSV", spreadsheet_safe_csv(exported), file_name=f"geovision_{hazard_type}_not_assessed.csv", mime="text/csv", key=f"hazard_csv_{hazard_type}")
    b.download_button("Analysis JSON", json.dumps(report, indent=2, allow_nan=False), file_name=f"geovision_{hazard_type}_analysis_notes.json", mime="application/json", key=f"hazard_manifest_{hazard_type}")


def render_hazard_results(
    result, run_info, status, *, hazard_label, hazard_type, requested_mode,
    selected_areas, service_radius_km, all_towers, pop_grid, area_id,
    simulate_coverage, base_map, add_track, map_config, land_cover_classes, completed_at,
) -> None:
    context = build_result_context(result, run_info, requested_mode, completed_at=completed_at)
    _evidence_strip(context)
    if not context["assessment"]["score_allowed"]:
        _render_unassessed(result, context, hazard_type, selected_areas, base_map, map_config)
        _source_details(context, status)
        return
    hz = result.copy().sort_values(["hazard_ai_score", "tower_id"], ascending=[False, True])
    st.markdown("#### Population impact")
    scenario_enabled = st.checkbox("Simulate tower outages", value=True, key=f"hazard_scenario_enabled_{hazard_type}",
                                   help="Hypothetical outages based on the exposure threshold.")
    metrics = None
    failure_threshold = None
    hz["assumed_unavailable"] = pd.Series(pd.NA, index=hz.index, dtype="boolean")
    if scenario_enabled:
        with st.expander("Scenario settings", expanded=False):
            failure_threshold = st.slider(
                "Outage threshold",
                min_value=.30, max_value=.90, value=float(status["high_threshold"]), step=.05,
                key=f"hazard_ai_failure_threshold_{hazard_type}",
                help="Assume a tower is unavailable when its exposure score meets this threshold. Scores stay unchanged.",
            )
        all_risk = all_towers.copy()
        all_risk["scenario_risk"] = all_risk.tower_id.map(hz.set_index("tower_id").hazard_ai_score).fillna(0.0)
        all_risk["risk_class"] = "Not assessed"
        metrics, loads = simulate_coverage(pop_grid, all_risk, [area_id[a] for a in selected_areas], selected_areas, service_radius_km, failure_threshold)
        hz = attach_coverage_scenario(hz, loads)
        hz["assumed_unavailable"] = hz.hazard_ai_score.ge(failure_threshold)
        st.caption(f"Simulation · {service_radius_km:g} km radius · threshold ≥ {failure_threshold:.2f} · 2020 population")
        m2, m3, m4, m1, m5 = st.columns(5)
        m2.metric("Affected people", f"{metrics['population_directly_affected']:,.0f}")
        m3.metric("Coverage loss", f"{metrics['population_losing_coverage']:,.0f}")
        m4.metric("Rerouted", f"{metrics['population_rerouted']:,.0f}")
        m1.metric("Offline towers", f"{metrics['selected_failed_towers']:,}")
        m5.metric("Coverage left", f"{metrics['post_coverage_pct']:.1f}%")

    options = ["Coverage loss", "Exposure score"] if scenario_enabled else ["Exposure score"]
    order = st.selectbox("Sort by", options, key=f"hazard_review_order_{hazard_type}_{scenario_enabled}")
    sort_columns = ["coverage_impact", "hazard_ai_score", "tower_id"] if order == "Coverage loss" else ["hazard_ai_score", "coverage_impact", "tower_id"]
    queue = hz.sort_values(sort_columns, ascending=[False, False, True])
    map_column, review_column = st.columns([2.1, 1], gap="large")
    with map_column:
        st.markdown("#### Exposure map")
        render_map(_exposure_map(hz, hazard_type, selected_areas, run_info, base_map, add_track, scenario_enabled), width="stretch", config=map_config, key=f"hazard_exposure_map_{hazard_type}")
        st.caption(f"{len(hz):,} towers · exposure 0–100")
    with review_column:
        st.markdown("#### Priority towers")
        for rank, row in enumerate(queue.head(3).itertuples(), 1):
            with st.container(border=True):
                st.markdown(f"**{rank}. Tower {row.tower_id} · {row.adm3_name}**")
                st.caption(f"{row.hazard_ai_score * 100:.1f} / 100 · {row.risk_level} exposure")
                if scenario_enabled:
                    st.write(f"**{row.coverage_impact:,.0f}** losing coverage")
                elif hasattr(row, "background_reference_score"):
                    st.write(f"Zero-event reference: {row.background_reference_score * 100:.1f} / 100")

    st.markdown("#### Top 20 towers")
    queue_columns = ["tower_id", "adm3_name", "hazard_ai_pct", "risk_level", "assumed_unavailable", "coverage_impact"]
    if not scenario_enabled:
        queue_columns = queue_columns[:4]
    rename = {
        "tower_id": "Tower ID", "adm3_name": "Township", "analysis_area": "Analysis area",
        "hazard_ai_pct": "Exposure score (0–100)", "risk_level": "Exposure band",
        "assumed_unavailable": "Assumed unavailable", "coverage_impact": "Losing coverage (scenario)",
        "population_affected": "Initially affected (scenario)", "population_rerouted": "Rerouted (scenario)",
        "baseline_population_served": f"People served before scenario ({service_radius_km:g} km)",
        "recommended_action": "Suggested engineering review", "main_factors": "Model explanation (TreeSHAP)",
        "background_reference_score": "Zero-event reference (0–1)",
        "event_score_delta": "Event-input change (score units)",
    }
    shown = queue[queue_columns].head(20).round({"hazard_ai_pct": 1, "coverage_impact": 0}).rename(columns=rename)
    st.dataframe(shown, width="stretch", hide_index=True, height=300)

    with st.expander("Tower details", expanded=False):
        explain_id = st.selectbox("Tower", queue.tower_id.tolist(), key=f"hazard_explanation_tower_{hazard_type}", format_func=lambda value: f"Tower {value}")
        explained = hz.set_index("tower_id").loc[explain_id]
        st.write(f"**Tower {explain_id} · {explained.adm3_name}** — exposure {explained.hazard_ai_score * 100:.1f} / 100")
        action = str(explained.recommended_action).removeprefix("Planning guidance: ").split(";")[0]
        action = action.replace("Prioritize an engineering review of", "Review").replace("Plan an engineering review of", "Review")
        st.write(action.rstrip(".") + ".")
        st.caption("TreeSHAP: " + str(explained.main_factors))

    _source_details(context, status)

    with st.expander("Diagnostics", expanded=False):
        if {"background_reference_score", "event_score_delta"}.issubset(hz.columns):
            a, b, c = st.columns(3)
            a.metric("Median model score", f"{hz.hazard_ai_score.median() * 100:.1f} / 100")
            b.metric("Median zero-event reference", f"{hz.background_reference_score.median() * 100:.1f} / 100")
            c.metric("Median event-input change", f"{hz.event_score_delta.median() * 100:+.1f} points")
            st.caption("Reference: event_intensity = 0. Delta measures model sensitivity.")
        risk_counts = hazard_summary(hz).set_index("risk_level").reindex(["Low", "Moderate", "High", "Very High"]).reset_index()
        chart = go.Figure(go.Bar(x=risk_counts.risk_level, y=risk_counts.tower_count,
                                marker_color=[RISK_COLORS[level] for level in risk_counts.risk_level],
                                hovertemplate="%{x} exposure: %{y:,} towers<extra></extra>"))
        chart.update_layout(height=220, margin={"l": 0, "r": 0, "t": 10, "b": 0}, yaxis_title="Tower count")
        st.plotly_chart(chart, width="stretch", key=f"hazard_risk_summary_{hazard_type}")
        details = queue.copy().head(100)
        if "land_cover" in details:
            details["land_cover"] = pd.to_numeric(details.land_cover, errors="coerce").astype("Int64").map(land_cover_classes).fillna("Unknown / low confidence")
        display_columns = [*rename, "networks", "radios", "rain_24h_mm", "rain_72h_mm", "rain_30d_mm", "elevation_m", "slope_deg", "land_cover", "storm_id", "storm_name", "track_status", "wind_speed", "pressure", "distance_to_cyclone", "magnitude", "depth", "distance_from_epicenter", "flood_ai_score", "earthquake_ai_score", "cyclone_ai_score", "hazard_data_source", "hazard_data_timestamp"]
        if not scenario_enabled:
            display_columns = [name for name in display_columns if name not in {
                "assumed_unavailable", "coverage_impact", "population_affected",
                "population_rerouted", "baseline_population_served",
            }]
        details = details[[name for name in display_columns if name in details]].replace([np.inf, -np.inf], np.nan)
        st.caption("Top 100 towers · component scores: 0–1 · exposure: 0–100")
        st.dataframe(details.rename(columns=rename), width="stretch", hide_index=True, height=400)

    if scenario_enabled:
        exported, report = build_scenario_report(hz, context, metrics, areas=selected_areas, radius_km=service_radius_km, threshold=failure_threshold)
    else:
        exported = annotate_result_exports(hz, context)
        report = {
            "report_version": "GeoVision v21", "result_context": context,
            "scenario": {"enabled": False}, "assessed_tower_count": len(hz),
            "observed_outage_validation": "Not available; models use proxy training labels",
        }
    report["model"] = {
        "name": status["model_name"], "type": status["model_type"],
        "features_in_order": status["features"], "training_rows": status["training_rows"],
        "training_towers": status["training_towers"],
        "validation_interpretation": "Proxy-formula reproduction only; no observed outage accuracy established",
    }
    st.markdown("#### Export")
    export_col, report_col = st.columns(2)
    export_col.download_button("Results CSV", spreadsheet_safe_csv(exported), file_name=f"geovision_{hazard_type}_scenario.csv", mime="text/csv", key=f"hazard_csv_{hazard_type}")
    report_col.download_button("Analysis JSON", json.dumps(report, indent=2, allow_nan=False), file_name=f"geovision_{hazard_type}_analysis_notes.json", mime="application/json", key=f"hazard_manifest_{hazard_type}")
