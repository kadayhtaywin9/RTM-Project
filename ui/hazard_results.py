"""Map-first hazard results with explicit evidence and scenario interpretation."""
from __future__ import annotations

import json
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.decision_engine import attach_coverage_scenario
from utils.result_context import build_result_context
from utils.scenario_report import build_scenario_report, spreadsheet_safe_csv
from utils.visualization import RISK_COLORS, hazard_summary


def _evidence_strip(context: dict) -> None:
    status = context["evidence_status"]
    label = escape(context["evidence_label"])
    summary = escape(context["evidence_summary"])
    completed = context.get("completed_at")
    completed_text = pd.Timestamp(completed).strftime("%d %b %Y · %H:%M UTC") if completed else "Time not recorded"
    st.markdown(
        f'<section class="gv-evidence gv-evidence-{escape(status)}" aria-label="Result evidence">'
        f'<div><span class="gv-evidence-label">{label}</span><span class="gv-research-label">Research model</span></div>'
        f'<p>{summary}</p><small>Analysis completed {escape(completed_text)} · Not an observation timestamp</small></section>',
        unsafe_allow_html=True,
    )
    st.caption("Exposure scores rank planning priorities. They are not probabilities of disaster, tower failure or measured outages.")


def _exposure_map(hz, hazard_type, selected_areas, run_info, base_map, add_track):
    fig = base_map(selected_areas, hz, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    if hazard_type == "cyclone":
        add_track(fig, run_info.get("forecast_track", []))
    customdata = np.column_stack([
        hz.tower_id, hz.adm3_name, hz.hazard_ai_score * 100, hz.risk_level,
        hz.coverage_impact, hz.assumed_unavailable.map({True: "Yes", False: "No"}),
    ])
    # Category boundaries exactly match the classifier's 30/50/70 thresholds.
    colorscale = [
        [0, RISK_COLORS["Low"]], [.299999, RISK_COLORS["Low"]],
        [.30, RISK_COLORS["Moderate"]], [.499999, RISK_COLORS["Moderate"]],
        [.50, RISK_COLORS["High"]], [.699999, RISK_COLORS["High"]],
        [.70, RISK_COLORS["Very High"]], [1, RISK_COLORS["Very High"]],
    ]
    fig.add_trace(go.Scattermap(
        lat=hz.lat, lon=hz.lon, mode="markers",
        marker={"size": np.where(hz.assumed_unavailable, 9, 6), "opacity": .8,
                "color": hz.hazard_ai_score * 100, "cmin": 0, "cmax": 100,
                "colorscale": colorscale, "showscale": True,
                "colorbar": {"title": {"text": "Exposure<br>0–100"}, "thickness": 10, "len": .65}},
        customdata=customdata,
        hovertemplate=("<b>Tower %{customdata[0]}</b><br>%{customdata[1]}"
                       "<br>Exposure score: %{customdata[2]:.1f} / 100 · %{customdata[3]}"
                       "<br>Assumed unavailable: %{customdata[5]}"
                       "<br>People losing coverage in scenario: %{customdata[4]:,.0f}"
                       "<extra>Planning scenario — not observed outages</extra>"),
        name="Tower exposure",
    ))
    fig.update_layout(height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False)
    return fig


def render_hazard_results(
    result, run_info, status, *, hazard_label, hazard_type, requested_mode,
    selected_areas, service_radius_km, all_towers, pop_grid, area_id,
    simulate_coverage, base_map, add_track, map_config, land_cover_classes, completed_at,
) -> None:
    context = build_result_context(result, run_info, requested_mode, completed_at=completed_at)
    _evidence_strip(context)
    hz = result.copy().sort_values(["hazard_ai_score", "tower_id"], ascending=[False, True])
    with st.expander("Scenario assumptions", expanded=False):
        st.caption("This is a what-if test. Moving the threshold does not change the AI scores or establish that a tower has failed.")
        failure_threshold = st.slider(
            "Treat tower as unavailable at exposure score",
            min_value=.30, max_value=.90, value=float(status["high_threshold"]), step=.05,
            key=f"hazard_ai_failure_threshold_{hazard_type}",
            help="A score of 0.50 is 50 / 100, not a 50% failure probability.",
        )
        st.write(f"Service radius: **{service_radius_km:g} km**. Change it in the sidebar.")
        st.caption("All surviving towers can receive rerouted population. Unassessed towers outside the selected area are assumed available. No RF, capacity, power or backhaul constraints are simulated.")
    all_risk = all_towers.copy()
    all_risk["scenario_risk"] = all_risk.tower_id.map(hz.set_index("tower_id").hazard_ai_score).fillna(0.0)
    all_risk["risk_class"] = "Not assessed"
    metrics, loads = simulate_coverage(pop_grid, all_risk, [area_id[a] for a in selected_areas], selected_areas, service_radius_km, failure_threshold)
    hz = attach_coverage_scenario(hz, loads)
    hz["assumed_unavailable"] = hz.hazard_ai_score.ge(failure_threshold)

    st.markdown("#### If these towers become unavailable")
    st.caption(f"{service_radius_km:g} km radius · exposure threshold {failure_threshold * 100:.0f} / 100 · 2020 population estimates, not observed affected people")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Assumed unavailable", f"{metrics['selected_failed_towers']:,}", help=f"Out of {len(hz):,} assessed towers; not confirmed failures.")
    m2.metric("People losing coverage", f"{metrics['population_losing_coverage']:,.0f}", help="Previously within range, now outside every surviving tower's range in this scenario.")
    m3.metric("People rerouted", f"{metrics['population_rerouted']:,.0f}", help="Still within another surviving tower's radius; capacity is not checked.")
    m4.metric("Coverage after scenario", f"{metrics['post_coverage_pct']:.1f}%",
              help=f"Baseline: {metrics['baseline_coverage_pct']:.1f}%. Scenario change: {metrics['post_coverage_pct'] - metrics['baseline_coverage_pct']:+.1f} percentage points.")
    st.caption(f"Population accounting: {metrics['population_directly_affected']:,.0f} initially affected = {metrics['population_rerouted']:,.0f} rerouted + {metrics['population_losing_coverage']:,.0f} losing coverage (rounded for display).")

    order = st.selectbox("Review priority", ["People losing coverage in this scenario", "Exposure score"], key=f"hazard_review_order_{hazard_type}")
    sort_columns = ["coverage_impact", "hazard_ai_score", "tower_id"] if order.startswith("People") else ["hazard_ai_score", "coverage_impact", "tower_id"]
    queue = hz.sort_values(sort_columns, ascending=[False, False, True])
    map_column, review_column = st.columns([2.1, 1], gap="large")
    with map_column:
        st.markdown("#### Exposure map")
        st.plotly_chart(_exposure_map(hz, hazard_type, selected_areas, run_info, base_map, add_track), width="stretch", config=map_config, key=f"hazard_exposure_map_{hazard_type}")
        st.caption(f"All {len(hz):,} assessed tower sites are included. Larger points are assumed unavailable in this scenario. Colors show exposure bands, not failure likelihood.")
    with review_column:
        st.markdown("#### Where to review first")
        st.caption("Screening shortlist for an engineer—not a dispatch instruction.")
        for rank, row in enumerate(queue.head(3).itertuples(), 1):
            with st.container(border=True):
                st.markdown(f"**{rank}. Tower {row.tower_id} · {row.adm3_name}**")
                st.caption(f"{row.hazard_ai_score * 100:.1f} / 100 · {row.risk_level} exposure")
                st.write(f"**{row.coverage_impact:,.0f} people** losing coverage in this scenario")
        st.caption("Use the tower details below for full review guidance and model explanations.")

    st.markdown("#### Tower review queue")
    st.caption("Top 20 under the selected review order. Full assessed results and scenario assumptions are available below.")
    queue_columns = ["tower_id", "adm3_name", "hazard_ai_pct", "risk_level", "assumed_unavailable", "coverage_impact"]
    rename = {
        "tower_id": "Tower ID", "adm3_name": "Township", "analysis_area": "Analysis area",
        "hazard_ai_pct": "Exposure score (0–100)", "risk_level": "Exposure band",
        "assumed_unavailable": "Assumed unavailable", "coverage_impact": "Losing coverage (scenario)",
        "population_affected": "Initially affected (scenario)", "population_rerouted": "Rerouted (scenario)",
        "baseline_population_served": f"People served before scenario ({service_radius_km:g} km)",
        "recommended_action": "Suggested engineering review", "main_factors": "Model explanation (TreeSHAP)",
    }
    shown = queue[queue_columns].head(20).round({"hazard_ai_pct": 1, "coverage_impact": 0}).rename(columns=rename)
    st.dataframe(shown, width="stretch", hide_index=True, height=300)

    with st.expander("Inspect a tower and its model explanation", expanded=False):
        explain_id = st.selectbox("Tower to inspect", queue.tower_id.tolist(), key=f"hazard_explanation_tower_{hazard_type}", format_func=lambda value: f"Tower {value}")
        explained = hz.set_index("tower_id").loc[explain_id]
        st.write(f"**Tower {explain_id} · {explained.adm3_name}** — exposure {explained.hazard_ai_score * 100:.1f} / 100")
        st.write(explained.recommended_action)
        st.caption(explained.main_factors)
        st.caption("TreeSHAP explains this model's score, not real-world causation. Positive/negative contributions move the score up/down; they are not percentages of physical impact.")

    with st.expander("Source evidence and model limitations", expanded=False):
        source_table = pd.DataFrame([{
            "Hazard": item["hazard_type"].title(), "Evidence": item["evidence_label"], "Source": item["source"],
            "Event / product time": " | ".join(item["data_timestamps"]) or "Not recorded",
            "Source checked at": item.get("source_checked_at") or "Not recorded",
            "Timestamp coverage": item["timestamp_status"].replace("_", " "),
        } for item in context["source_records"]])
        st.dataframe(source_table, width="stretch", hide_index=True)
        st.caption("Event/product time and fetch time mean different things. Compound inputs retain separate clocks; no single timestamp establishes freshness of all sources.")
        if hazard_type == "earthquake" and context["evidence_status"] == "live":
            st.write(f"USGS: {run_info.get('event_count', 0):,} events in the {run_info.get('window_days', 30)}-day query window. This is post-event screening, not earthquake prediction.")
        if hazard_type == "cyclone" and context["evidence_status"] == "live":
            st.write(f"JTWC: {run_info.get('active_storm_count', 0):,} active storms in the queried basins; {run_info.get('track_points', 0):,} current/forecast points.")
        for limitation in dict.fromkeys(context["limitations"] + status["limitations"]):
            st.write(f"- {limitation}")

    with st.expander("Exposure distribution and detailed tower data", expanded=False):
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
        details = details[[name for name in display_columns if name in details]].replace([np.inf, -np.inf], np.nan)
        st.caption("Top 100 towers. Component scores use 0–1; the headline exposure score uses 0–100. Missing context is unknown, not zero.")
        st.dataframe(details.rename(columns=rename), width="stretch", hide_index=True, height=400)

    exported, report = build_scenario_report(hz, context, metrics, areas=selected_areas, radius_km=service_radius_km, threshold=failure_threshold)
    report["model"] = {
        "name": status["model_name"], "type": status["model_type"],
        "features_in_order": status["features"], "training_rows": status["training_rows"],
        "training_towers": status["training_towers"],
        "validation_interpretation": "Proxy-formula reproduction only; no observed outage accuracy established",
    }
    st.markdown("#### Export this analysis")
    st.caption("Downloads include all assessed towers, evidence status, source timestamps and scenario assumptions. These are planning outputs, not observed impact reports.")
    export_col, report_col = st.columns(2)
    export_col.download_button("Download tower results (CSV)", spreadsheet_safe_csv(exported), file_name=f"geovision_{hazard_type}_scenario.csv", mime="text/csv", key=f"hazard_csv_{hazard_type}")
    report_col.download_button("Download scenario notes (JSON)", json.dumps(report, indent=2, allow_nan=False), file_name=f"geovision_{hazard_type}_scenario_notes.json", mime="application/json", key=f"hazard_manifest_{hazard_type}")
