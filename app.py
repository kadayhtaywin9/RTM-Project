from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from geoai_engine import (
    attach_rainfall_to_towers,
    coverage_summary,
    outage_risk,
    recommend_sites,
    simulate_population_coverage,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

YANGON_CITY_TOWNSHIPS = [
    "Latha", "Lanmadaw", "Pabedan", "Kyauktada", "Botahtaung", "Pazundaung",
    "Dagon", "Bahan", "Kamaryut", "Ahlone", "Kyeemyindaing", "Sanchaung",
    "Mingalartaungnyunt", "Tamwe", "Hlaing", "Thingangyun", "Yankin",
    "Dawbon", "Thaketa", "Insein", "Mayangone", "Dala", "Dagon Myothit (North)",
    "South Okkalapa", "North Okkalapa", "Hlaingtharya (East)", "Hlaingtharya (West)",
    "Shwepyithar", "Mingaladon", "Dagon Myothit (South)", "Dagon Myothit (East)",
    "Dagon Myothit (Seikkan)", "Seikgyikanaungto",
]
AREA_TOWNSHIPS = {
    "Yangon City": YANGON_CITY_TOWNSHIPS,
    "Hmawbi": ["Hmawbi"],
    "Thanlyin": ["Thanlyin"],
    "Kyauktan": ["Kyauktan"],
}
ANALYSIS_AREAS = list(AREA_TOWNSHIPS)
AREA_ID = {"Yangon City": 1, "Hmawbi": 2, "Thanlyin": 3, "Kyauktan": 4}
TOWNSHIP_TO_AREA = {t: a for a, towns in AREA_TOWNSHIPS.items() for t in towns}

st.set_page_config(page_title="Yangon GeoAI Telecom Resilience", page_icon="📡", layout="wide")


@st.cache_data
def load_json(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str):
    return pd.read_csv(DATA / name)


@st.cache_resource
def load_population_grid():
    z = np.load(DATA / "population_service_grid.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


admin3_geo = load_json("yangon_admin3.geojson")
flood_geo = load_json("historic_flood_analysis.geojson")
meta = load_json("metadata.json")
candidates_all = load_csv("candidate_village_tracts.csv")
tower_sites_all = load_csv("tower_sites_population.csv")
rainfall = load_csv("yangon_rainfall_5y.csv")
earthquakes = load_csv("earthquakes_near_yangon.csv")
cyclones = load_csv("cyclone_labels.csv")
pop_grid = load_population_grid()
rainfall["date"] = pd.to_datetime(rainfall["date"])


def expand_areas(areas):
    return sorted({township for area in areas for township in AREA_TOWNSHIPS[area]})


def filter_geojson(geo, prop, allowed):
    allowed = set(allowed)
    return {
        "type": "FeatureCollection",
        "features": [f for f in geo.get("features", []) if f.get("properties", {}).get(prop) in allowed],
    }


def map_center(df):
    if len(df):
        return {"lat": float(df.lat.mean()), "lon": float(df.lon.mean())}
    return {"lat": 16.86, "lon": 96.20}


def base_map(selected_areas, points_df=None, zoom=8.5):
    selected_townships = expand_areas(selected_areas)
    boundary = filter_geojson(admin3_geo, "adm3_name", selected_townships)
    fig = go.Figure()
    if boundary.get("features"):
        locs = [f["properties"]["adm3_name"] for f in boundary["features"]]
        fig.add_trace(
            go.Choroplethmap(
                geojson=boundary,
                locations=locs,
                z=[1] * len(locs),
                featureidkey="properties.adm3_name",
                colorscale=[[0, "rgba(20,90,160,0.10)"], [1, "rgba(20,90,160,0.10)"]],
                marker={"line": {"width": 1.2}},
                showscale=False,
                hovertemplate="<b>%{location}</b><extra></extra>",
                name="Analysis boundary",
            )
        )
    default_points = candidates_all[candidates_all.adm3_name.isin(selected_townships)]
    center = map_center(points_df if points_df is not None and len(points_df) else default_points)
    fig.update_layout(
        map={"style": "open-street-map", "center": center, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=580,
        legend={"orientation": "h", "y": 0.01, "x": 0.01},
    )
    return fig


def add_towers(fig, towers, max_points=2500):
    draw = towers.copy()
    if len(draw) > max_points:
        draw = draw.sample(max_points, random_state=42)
    fig.add_trace(
        go.Scattermap(
            lat=draw.lat,
            lon=draw.lon,
            mode="markers",
            marker={"size": 5, "color": "#335CFF", "opacity": 0.5},
            customdata=np.column_stack([
                draw.adm3_name,
                draw.radios,
                draw.cell_count,
                draw.estimated_population_nearest,
            ]),
            hovertemplate=(
                "<b>Tower-site proxy</b><br>Township: %{customdata[0]}"
                "<br>Radio: %{customdata[1]}<br>Observed cells: %{customdata[2]}"
                "<br>Estimated nearest-catchment population: %{customdata[3]:,.0f}<extra></extra>"
            ),
            name="Observed tower sites",
        )
    )
    return fig


def add_candidates(fig, df, name="Admin-4 areas"):
    if df.empty:
        return fig
    size = 7 + 9 * df.gap_score.clip(0, 1)
    fig.add_trace(
        go.Scattermap(
            lat=df.lat,
            lon=df.lon,
            mode="markers",
            marker={
                "size": size,
                "color": df.nearest_tower_km,
                "colorscale": "YlOrRd",
                "showscale": True,
                "colorbar": {"title": "Gap km", "x": 0.98},
            },
            customdata=np.column_stack([
                df.adm3_name,
                df.adm4_name,
                df.nearest_tower_km,
                df.is_rural,
                df.population_2020,
            ]),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>%{customdata[0]}"
                "<br>Nearest observed site: %{customdata[2]:.1f} km"
                "<br>Rural flag: %{customdata[3]}"
                "<br>Population 2020: %{customdata[4]:,.0f}<extra></extra>"
            ),
            name=name,
        )
    )
    return fig


def add_recommendations(fig, recs):
    if recs.empty:
        return fig
    fig.add_trace(
        go.Scattermap(
            lat=recs.lat,
            lon=recs.lon,
            mode="markers+text",
            text=[str(x) for x in recs["rank"]],
            textposition="top center",
            marker={"size": 16, "color": "#00A878", "opacity": 0.95},
            customdata=np.column_stack([
                recs.adm3_name,
                recs.adm4_name,
                recs.nearest_tower_km,
                recs.population_2020,
                recs.suitability_score,
            ]),
            hovertemplate=(
                "<b>Recommended site #%{text}</b><br>%{customdata[1]}, %{customdata[0]}"
                "<br>Current gap: %{customdata[2]:.1f} km"
                "<br>Admin-4 population: %{customdata[3]:,.0f}"
                "<br>Suitability: %{customdata[4]:.1f}/100<extra></extra>"
            ),
            name="Recommended new sites",
        )
    )
    return fig


def add_flood_layer(fig):
    feats = flood_geo.get("features", [])
    if not feats:
        return fig
    locs = [f.get("properties", {}).get("id", str(i)) for i, f in enumerate(feats)]
    z = [f.get("properties", {}).get("flood_frequency", 0) for f in feats]
    fig.add_trace(
        go.Choroplethmap(
            geojson=flood_geo,
            locations=locs,
            z=z,
            featureidkey="properties.id",
            colorscale=[[0, "rgba(180,220,255,0.15)"], [1, "rgba(20,90,190,0.60)"]],
            marker={"line": {"width": 0}},
            showscale=True,
            colorbar={"title": "Historic flood frequency"},
            hovertemplate="Flood frequency: %{z}<extra></extra>",
            name="Historic flood footprint",
        )
    )
    return fig


def add_area_label(df):
    out = df.copy()
    out["analysis_area"] = out["adm3_name"].map(TOWNSHIP_TO_AREA).fillna(out["adm3_name"])
    return out


def selected_population_mask(selected_areas):
    return np.isin(pop_grid["area_id"], [AREA_ID[a] for a in selected_areas])


def area_summary(df, selected_areas, threshold):
    rows = []
    for area in selected_areas:
        subset = df[df.adm3_name.isin(AREA_TOWNSHIPS[area])]
        if subset.empty:
            continue
        underserved = subset.nearest_tower_km >= threshold
        rows.append({
            "Analysis area": area,
            "Admin-4 units": int(len(subset)),
            "Population 2020": int(round(subset.population_2020.sum())),
            "Median gap (km)": round(float(subset.nearest_tower_km.median()), 2),
            "Max gap (km)": round(float(subset.nearest_tower_km.max()), 2),
            "Underserved units": int(underserved.sum()),
            "Population in underserved units": int(round(subset.loc[underserved, "population_2020"].sum())),
        })
    return pd.DataFrame(rows)


# ---------- sidebar ----------
st.sidebar.title("GeoAI controls")
selected_areas = st.sidebar.multiselect(
    "Analysis areas",
    options=ANALYSIS_AREAS,
    default=ANALYSIS_AREAS,
    help="Yangon City uses the 33-township YCDC area; Hmawbi, Thanlyin and Kyauktan are analyzed separately.",
)
if not selected_areas:
    selected_areas = ANALYSIS_AREAS

selected_townships = expand_areas(selected_areas)
service_radius_km = st.sidebar.slider(
    "Planning service radius (km)", 1.0, 15.0, 5.0, 0.5,
    help="Used only for population coverage simulation. It is not an RF propagation radius.",
)
threshold_km = st.sidebar.slider("Underserved gap threshold (km)", 2.0, 20.0, 5.0, 0.5)
n_sites = st.sidebar.slider("Recommended new sites", 3, 20, 10)
min_spacing = st.sidebar.slider("Minimum spacing between new sites (km)", 2.0, 20.0, 8.0, 1.0)
st.sidebar.markdown("**Tower suitability weights**")
w_gap = st.sidebar.slider("Coverage gap", 0.0, 1.0, 0.45, 0.05)
w_pop = st.sidebar.slider("Population demand", 0.0, 1.0, 0.30, 0.05)
w_rural = st.sidebar.slider("Rural priority", 0.0, 1.0, 0.15, 0.05)
w_safe = st.sidebar.slider("Hazard safety", 0.0, 1.0, 0.10, 0.05)

candidates = candidates_all[candidates_all.adm3_name.isin(selected_townships)].copy()
towers = tower_sites_all[tower_sites_all.adm3_name.isin(selected_townships)].copy()
summary = coverage_summary(candidates, threshold_km)
summary_area = area_summary(candidates, selected_areas, threshold_km)
recs = recommend_sites(candidates, n_sites, min_spacing, w_gap, w_pop, w_rural, w_safe)

# Baseline population coverage with no simulated failures.
base_towers = tower_sites_all.copy()
base_towers["scenario_risk"] = 0.0
baseline_metrics, baseline_load = simulate_population_coverage(
    pop_grid,
    base_towers,
    [AREA_ID[a] for a in selected_areas],
    selected_areas,
    service_radius_km,
    2.0,
)

# ---------- header ----------
st.title("📡 GeoAI Population, Connectivity & Disaster-Resilient Telecom Dashboard")
st.caption(
    "Yangon City + Hmawbi + Thanlyin + Kyauktan • population demand + tower load + flood/rainfall + disaster coverage simulation"
)
st.info(
    "Population per tower is an **estimated geographic service population**, not a subscriber count. "
    "WorldPop pixels are assigned to their nearest observed tower-site proxy; the planning service radius controls whether a pixel is counted as covered."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Population 2020", f"{baseline_metrics.get('population_total', 0):,.0f}")
k2.metric("Observed tower-site proxies", f"{len(towers):,}")
k3.metric("Baseline covered population", f"{baseline_metrics.get('baseline_served', 0):,.0f}", f"{baseline_metrics.get('baseline_coverage_pct', 0):.1f}%")
k4.metric("Baseline uncovered", f"{baseline_metrics.get('baseline_uncovered', 0):,.0f}", f"> {service_radius_km:.1f} km")
k5.metric("Underserved Admin-4 units", f"{int((candidates.nearest_tower_km >= threshold_km).sum()):,}")

# ---------- tabs ----------
t_overview, t_population, t_gap, t_recommend, t_disaster, t_rain, t_method = st.tabs(
    [
        "Overview",
        "Population & tower load",
        "Underserved areas",
        "Tower recommendations",
        "Disaster coverage",
        "Rainfall & flood",
        "Data & method",
    ]
)

with t_overview:
    st.subheader("Regional network picture")
    fig = base_map(selected_areas, towers, zoom=8.45 if len(selected_areas) > 1 else 9.1)
    add_towers(fig, towers)
    add_candidates(fig, candidates, "Admin-4 gap points")
    add_recommendations(fig, recs)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Analysis-area gap + population summary")
    st.dataframe(summary_area, use_container_width=True, hide_index=True)

with t_population:
    st.subheader("How many people are associated with each observed tower site?")
    st.write(
        "Each WorldPop 2020 raster pixel is assigned to its nearest observed tower-site proxy. "
        "This creates a Voronoi-like geographic catchment and gives an estimated population load. "
        "It does not tell us actual SIM subscribers, traffic, handovers or sector utilization."
    )

    p_rows = []
    for area in selected_areas:
        area_towers = tower_sites_all[tower_sites_all.analysis_area == area]
        aid = AREA_ID[area]
        pop_total = float(pop_grid["population"][pop_grid["area_id"] == aid].sum())
        p_rows.append({
            "Area": area,
            "Population 2020": int(round(pop_total)),
            "Tower-site proxies": int(len(area_towers)),
            "Population / site": int(round(pop_total / max(len(area_towers), 1))),
            "Median nearest-catchment population": int(round(area_towers.estimated_population_nearest.median())),
            "Max nearest-catchment population": int(round(area_towers.estimated_population_nearest.max())),
            "Sites inside historic flood footprint": int((area_towers.flood_frequency > 0).sum()),
        })
    st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1.55, 1])
    with c1:
        fig = base_map(selected_areas, towers, zoom=8.6 if len(selected_areas) > 1 else 9.2)
        draw = towers.copy()
        if len(draw) > 3000:
            high = draw.nlargest(700, "estimated_population_nearest")
            rest = draw.drop(high.index).sample(min(2300, max(0, len(draw) - len(high))), random_state=42)
            draw = pd.concat([high, rest]).drop_duplicates("tower_id")
        size = 5 + 10 * np.log1p(draw.estimated_population_nearest) / max(np.log1p(draw.estimated_population_nearest.max()), 1e-9)
        fig.add_trace(
            go.Scattermap(
                lat=draw.lat,
                lon=draw.lon,
                mode="markers",
                marker={
                    "size": size,
                    "color": np.log1p(draw.estimated_population_nearest),
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "log(population load)"},
                },
                customdata=np.column_stack([
                    draw.tower_id,
                    draw.adm3_name,
                    draw.radios,
                    draw.cell_count,
                    draw.estimated_population_nearest,
                    draw.estimated_population_primary_5km,
                    draw.flood_frequency,
                ]),
                hovertemplate=(
                    "<b>Site %{customdata[0]}</b><br>%{customdata[1]}"
                    "<br>Radio: %{customdata[2]} | cells: %{customdata[3]}"
                    "<br>Nearest-catchment population: %{customdata[4]:,.0f}"
                    "<br>Primary population within 5 km: %{customdata[5]:,.0f}"
                    "<br>Historic flood frequency: %{customdata[6]:.0f}<extra></extra>"
                ),
                name="Estimated tower population load",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Highest estimated population loads**")
        top = towers.nlargest(25, "estimated_population_primary_5km")[[
            "tower_id", "analysis_area", "adm3_name", "radios", "cell_count",
            "estimated_population_primary_5km", "estimated_population_nearest", "flood_frequency",
        ]].copy()
        top["estimated_population_primary_5km"] = top.estimated_population_primary_5km.round(0).astype(int)
        top["estimated_population_nearest"] = top.estimated_population_nearest.round(0).astype(int)
        top.columns = [
            "Site ID", "Area", "Township", "Radio", "Cells", "Primary pop ≤5 km",
            "Nearest-catchment pop", "Flood freq",
        ]
        st.dataframe(top, use_container_width=True, hide_index=True, height=520)

    st.caption(
        "Use 'Primary pop ≤5 km' for a conservative planning-load view. 'Nearest-catchment pop' assigns every person to the nearest observed site even when farther than 5 km, so it highlights infrastructure scarcity but is not a signal-coverage estimate."
    )

with t_gap:
    st.subheader("Identify underserved areas and underserved population")
    underserved = candidates[candidates.nearest_tower_km >= threshold_km].sort_values(
        ["population_2020", "nearest_tower_km"], ascending=False
    )
    underserved = add_area_label(underserved)
    c1, c2 = st.columns([1.55, 1])
    with c1:
        fig = base_map(selected_areas, underserved if len(underserved) else candidates, zoom=8.6 if len(selected_areas) > 1 else 9.2)
        add_candidates(fig, underserved if len(underserved) else candidates, "Underserved admin-4 units")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Population in underserved Admin-4 units", f"{underserved.population_2020.sum():,.0f}")
        top = underserved[[
            "analysis_area", "adm3_name", "adm4_name", "population_2020", "nearest_tower_km", "hazard_score"
        ]].head(25).copy()
        if len(top):
            top["population_2020"] = top.population_2020.round(0).astype(int)
            top["nearest_tower_km"] = top.nearest_tower_km.round(1)
            top["hazard_score"] = (100 * top.hazard_score).round(0).astype(int)
        top.columns = ["Area", "Township", "Admin-4", "Population 2020", "Nearest site km", "Hazard /100"]
        st.dataframe(top, use_container_width=True, hide_index=True, height=490)

    st.info(
        "This improves the old 'distance only' view: a 10 km gap with 20,000 people can now be distinguished from a 10 km gap with 500 people."
    )

with t_recommend:
    st.subheader("Population-aware GeoAI tower recommendations")
    st.write(
        "Candidate areas are ranked using coverage gap, population demand, rural priority and hazard safety, then a minimum-spacing rule prevents recommendations from clustering together."
    )
    fig = base_map(selected_areas, recs, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    add_towers(fig, towers, max_points=1600)
    add_recommendations(fig, recs)
    st.plotly_chart(fig, use_container_width=True)

    recs_display = add_area_label(recs)
    export_cols = [
        "rank", "analysis_area", "adm3_name", "adm4_name", "lat", "lon", "population_2020",
        "nearest_tower_km", "gap_score", "population_score", "hazard_score", "safety_score", "suitability_score",
    ]
    table = recs_display[export_cols].copy()
    for c in ["lat", "lon", "nearest_tower_km", "gap_score", "population_score", "hazard_score", "safety_score", "suitability_score"]:
        table[c] = table[c].round(4)
    table["population_2020"] = table.population_2020.round(0).astype(int)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download population-aware recommended sites CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="geoai_population_aware_recommended_sites.csv",
        mime="text/csv",
    )

with t_disaster:
    st.subheader("How a disaster changes population coverage")
    st.write(
        "The simulator first estimates site risk, assumes sites above your risk threshold are unavailable, then reassigns population pixels to the nearest surviving alternative among the 10 precomputed nearest sites. "
        "People are counted as losing coverage when no surviving alternative is within the planning service radius."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scenario = st.selectbox("Scenario", ["Flood / Heavy Rain", "Earthquake", "Cyclone", "Compound"])
    with c2:
        severity = st.slider("Scenario severity", 1, 5, 3)
    rain_dates = sorted(rainfall.date.dt.strftime("%Y-%m-%d").unique().tolist())
    with c3:
        rain_date = st.selectbox("Rainfall snapshot", rain_dates, index=len(rain_dates) - 1)
    with c4:
        risk_threshold = st.slider("Assume site unavailable at risk ≥", 0.30, 0.90, 0.60, 0.05)

    risk_all = attach_rainfall_to_towers(tower_sites_all, rainfall, rain_date)
    risk_all = outage_risk(risk_all, scenario, severity)
    metrics, load_after = simulate_population_coverage(
        pop_grid,
        risk_all,
        [AREA_ID[a] for a in selected_areas],
        selected_areas,
        service_radius_km,
        risk_threshold,
    )
    risk = risk_all[risk_all.analysis_area.isin(selected_areas)].sort_values("scenario_risk", ascending=False)

    a, b, c, d, e = st.columns(5)
    a.metric("Assumed unavailable sites", f"{metrics.get('selected_failed_towers', 0):,}")
    b.metric("Primary population affected", f"{metrics.get('population_directly_affected', 0):,.0f}")
    c.metric("Population rerouted", f"{metrics.get('population_rerouted', 0):,.0f}")
    d.metric("Population losing coverage", f"{metrics.get('population_losing_coverage', 0):,.0f}")
    e.metric(
        "Coverage after disaster",
        f"{metrics.get('post_coverage_pct', 0):.1f}%",
        f"{metrics.get('post_coverage_pct', 0)-metrics.get('baseline_coverage_pct', 0):+.1f} pp",
    )

    fig = base_map(selected_areas, risk, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    draw = risk.head(2500).copy()
    failed_draw = risk[risk.scenario_risk >= risk_threshold]
    if len(failed_draw):
        draw = pd.concat([draw, failed_draw]).drop_duplicates("tower_id")
    fig.add_trace(
        go.Scattermap(
            lat=draw.lat,
            lon=draw.lon,
            mode="markers",
            marker={
                "size": np.where(draw.scenario_risk >= risk_threshold, 11, 7),
                "color": draw.scenario_risk,
                "cmin": 0,
                "cmax": 1,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Scenario risk"},
            },
            customdata=np.column_stack([
                draw.tower_id,
                draw.adm3_name,
                draw.radios,
                draw.scenario_risk,
                draw.estimated_population_primary_5km,
                draw.flood_frequency,
                draw.get("r1h", pd.Series(0, index=draw.index)),
                draw.get("r1q", pd.Series(0, index=draw.index)),
            ]),
            hovertemplate=(
                "<b>Site %{customdata[0]}</b><br>%{customdata[1]} | %{customdata[2]}"
                "<br>Scenario risk: %{customdata[3]:.1%}"
                "<br>Primary population ≤5 km: %{customdata[4]:,.0f}"
                "<br>Historic flood frequency: %{customdata[5]:.0f}"
                "<br>1-month rain: %{customdata[6]:.1f} mm | anomaly: %{customdata[7]:.0f}%<extra></extra>"
            ),
            name="Scenario risk",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Highest-risk sites and their baseline population load**")
        risk_table = load_after[load_after.analysis_area.isin(selected_areas)].sort_values(
            ["failed_in_scenario", "scenario_risk", "baseline_people_within_radius"], ascending=False
        )[[
            "tower_id", "analysis_area", "adm3_name", "scenario_risk", "failed_in_scenario",
            "baseline_people_within_radius", "post_disaster_people_within_radius", "flood_frequency",
        ]].head(30).copy()
        risk_table["scenario_risk"] = (100 * risk_table.scenario_risk).round(1)
        for col in ["baseline_people_within_radius", "post_disaster_people_within_radius"]:
            risk_table[col] = risk_table[col].round(0).astype(int)
        st.dataframe(risk_table, use_container_width=True, hide_index=True, height=460)
    with c2:
        st.markdown("**Surviving sites absorbing the largest extra population load**")
        gain = load_after[
            load_after.analysis_area.isin(selected_areas) & (~load_after.failed_in_scenario)
        ].sort_values("load_change_people", ascending=False)[[
            "tower_id", "analysis_area", "adm3_name", "radios", "baseline_people_within_radius",
            "post_disaster_people_within_radius", "load_change_people", "load_ratio",
        ]].head(30).copy()
        for col in ["baseline_people_within_radius", "post_disaster_people_within_radius", "load_change_people"]:
            gain[col] = gain[col].round(0).astype(int)
        gain["load_ratio"] = gain.load_ratio.replace([np.inf, -np.inf], np.nan).round(2)
        st.dataframe(gain, use_container_width=True, hide_index=True, height=460)

    st.warning(
        "The disaster result is a scenario estimate, not a calibrated outage forecast. We do not have actual tower-failure labels, antenna propagation, backup-power status or capacity limits. "
        "The model is strongest as a prioritization tool: which sites and populations should planners investigate first?"
    )

with t_rain:
    st.subheader("Five-year rainfall context + historic flood exposure")
    st.write(
        "The rainfall file is a dekadal (10-day) WFP/CHIRPS subnational series. The dashboard uses the 1-month rolling rainfall percentile as an event-stress indicator and combines it with the historic flood footprint for the flood/heavy-rain scenario."
    )

    selected_adm2 = tower_sites_all[tower_sites_all.analysis_area.isin(selected_areas)].adm2_pcode.dropna().unique().tolist()
    r = rainfall[rainfall.PCODE.isin(selected_adm2)].copy()
    fig = go.Figure()
    for name, sub in r.groupby("adm2_name"):
        fig.add_trace(go.Scatter(x=sub.date, y=sub.r1h, mode="lines", name=f"{name}: 1-month rain"))
        fig.add_trace(go.Scatter(x=sub.date, y=sub.r1h_avg, mode="lines", line={"dash": "dot"}, name=f"{name}: long-term avg"))
    fig.update_layout(
        height=430,
        xaxis_title="Date",
        yaxis_title="Rainfall (mm, 1-month rolling)",
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    latest_date = rainfall.date.max()
    latest = rainfall[(rainfall.date == latest_date) & rainfall.PCODE.isin(selected_adm2)][[
        "adm2_name", "date", "rfh", "rfh_avg", "rfq", "r1h", "r1h_avg", "r1q", "r1h_percentile", "r3h", "r3q"
    ]].copy()
    latest["r1h_percentile"] = (100 * latest.r1h_percentile).round(0).astype(int)
    st.markdown(f"**Latest supplied rainfall snapshot: {latest_date.date()}**")
    st.dataframe(latest, use_container_width=True, hide_index=True)

    pmask = selected_population_mask(selected_areas)
    hist_flood_pop = float(pop_grid["population"][pmask & (pop_grid["flood_frequency"] > 0)].sum())
    selected_pop = float(pop_grid["population"][pmask].sum())
    flood_sites = int((towers.flood_frequency > 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Population inside historic flood footprint", f"{hist_flood_pop:,.0f}", f"{100*hist_flood_pop/max(selected_pop,1):.1f}%")
    c2.metric("Tower sites inside historic flood footprint", f"{flood_sites:,}")
    c3.metric("Historic flood polygons in analysis dataset", f"{meta.get('flood_feature_count', 0):,}")

    fig = base_map(selected_areas, towers, zoom=8.5 if len(selected_areas) > 1 else 9.1)
    add_flood_layer(fig)
    flooded_towers = towers[towers.flood_frequency > 0]
    if len(flooded_towers):
        fig.add_trace(
            go.Scattermap(
                lat=flooded_towers.lat,
                lon=flooded_towers.lon,
                mode="markers",
                marker={"size": 11, "color": "red"},
                customdata=np.column_stack([
                    flooded_towers.tower_id,
                    flooded_towers.adm3_name,
                    flooded_towers.flood_frequency,
                    flooded_towers.estimated_population_primary_5km,
                ]),
                hovertemplate=(
                    "<b>Flood-exposed site %{customdata[0]}</b><br>%{customdata[1]}"
                    "<br>Historic frequency: %{customdata[2]:.0f}"
                    "<br>Primary population ≤5 km: %{customdata[3]:,.0f}<extra></extra>"
                ),
                name="Tower inside historic flood footprint",
            )
        )
    st.plotly_chart(fig, use_container_width=True)

with t_method:
    st.subheader("Data inventory and methodology")
    drows = pd.DataFrame([
        ["WorldPop population GeoTIFF", meta.get("population_grid_points", 0), "Usable", "Population count per raster pixel; reference year 2020"],
        ["Yangon admin boundaries", meta.get("yangon_admin3_count", 0), "Usable", f"Valid-on {meta.get('boundary_valid_on', '')}"],
        ["Yangon tower cells", meta.get("yangon_tower_cell_count", 0), "Usable with caveat", "Coordinate-dedup creates tower-site proxies"],
        ["Historic flood polygons", meta.get("flood_feature_count", 0), "Usable", "Flood footprint/frequency inside analysis region"],
        ["Rainfall time series", meta.get("rainfall_rows_yangon", 0), "Usable", f"{meta.get('rainfall_date_min')} to {meta.get('rainfall_date_max')}"],
        ["Earthquakes near Yangon", meta.get("earthquake_rows_used", 0), "Usable", "Historical exposure index"],
        ["Cyclone labels", meta.get("cyclone_rows_used", 0), "Limited", "Only one uploaded cyclone record"],
    ], columns=["Layer", "Rows/pixels/features", "Status", "Use"])
    st.dataframe(drows, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **1. Estimated population per tower-site proxy**
        - Clip the WorldPop 2020 population raster to Yangon City, Hmawbi, Thanlyin and Kyauktan.
        - For every populated raster pixel, precompute the 10 nearest observed tower-site proxies.
        - The nearest site receives that pixel's population as its primary geographic catchment.
        - A pixel is considered covered only when the nearest available site is inside the selected planning service radius.

        **2. Population-aware tower recommendation**
        - Coverage-gap score: larger distance to the nearest observed site = higher need.
        - Population-demand score: log-normalized Admin-4 WorldPop total.
        - Rural-priority flag: prioritizes non-urban Admin-4 areas.
        - Hazard-safety score: favors relatively safer candidate areas.
        - Default suitability weights: 45% gap + 30% population + 15% rural + 10% safety.

        **3. Flood/heavy-rain hazard**
        - Historic flood polygons provide spatial susceptibility and flood frequency.
        - WFP/CHIRPS 1-month rainfall percentile provides the event-stress trigger for the chosen date.
        - Flood/rain hazard score = 70% historic flood susceptibility + 30% rainfall stress.

        **4. Disaster coverage simulation**
        - Scenario risk combines hazard exposure, severity, site isolation and a radio-technology vulnerability proxy.
        - Sites above the selected risk threshold are treated as unavailable for the scenario.
        - Population is re-routed to the nearest surviving alternative among its 10 nearest precomputed sites.
        - If no surviving alternative is inside the planning service radius, that population is counted as losing coverage.
        """
    )

    st.markdown("#### Critical interpretation")
    st.error(
        "Estimated population per site is NOT the number of real customers connected to that tower. Actual users require operator subscriber/traffic data. "
        "Likewise, the planning radius is NOT RF propagation. A production model should add antenna frequency, height, azimuth, transmit power, terrain/DEM, buildings, handover/traffic logs, backup power and tower outage history."
    )
