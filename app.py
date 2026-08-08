from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from site_ai import assess_site, model_status, recommend_sites_xgb

try:
    import folium
    from streamlit_folium import st_folium
    HAS_CLICK_MAP = True
except Exception:
    folium = None
    st_folium = None
    HAS_CLICK_MAP = False

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

# Map-only display settings: keep navigation tools available on hover, remove selection
# mode buttons that clutter the top-right corner, and keep the Plotly logo hidden.
MAP_PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}



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
tower_sites_lookup_all = load_csv("tower_sites_yangon_all.csv")
tower_cells_lookup_all = load_csv("yangon_tower_cells_with_site.csv")
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
        # Keep the basemap full-height. Put the legend inside the map instead of
        # creating a large empty strip underneath it. The semi-transparent dark
        # legend stays readable in Streamlit light and dark themes.
        margin={"l": 8, "r": 84, "t": 10, "b": 8},
        height=560,
        legend={
            "orientation": "h",
            "y": 0.015,
            "yanchor": "bottom",
            "x": 0.5,
            "xanchor": "center",
            "bgcolor": "rgba(20,24,32,0.76)",
            "bordercolor": "rgba(255,255,255,0.20)",
            "borderwidth": 1,
            "font": {"color": "white", "size": 11},
        },
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
                "colorbar": {
                    "title": {"text": "Gap km", "side": "right"},
                    "x": 1.02,
                    "xanchor": "left",
                    "y": 0.47,
                    "yanchor": "middle",
                    "len": 0.68,
                    "thickness": 16,
                    "outlinewidth": 0,
                },
            },
            # The first value is a stable tag used by Streamlit click selection.
            customdata=np.array(list(zip(
                np.repeat("gap", len(df)),
                df.candidate_id.astype(int),
                df.nearest_tower_id.astype(int),
                df.adm3_name.astype(str),
                df.adm4_name.fillna("Unnamed").astype(str),
                df.nearest_tower_km.astype(float),
                df.is_rural.astype(int),
                df.population_2020.astype(float),
            )), dtype=object),
            hovertemplate=(
                "<b>%{customdata[4]}</b><br>%{customdata[3]}"
                "<br>Nearest observed site: %{customdata[5]:.2f} km"
                "<br>Rural flag: %{customdata[6]}"
                "<br>Population 2020: %{customdata[7]:,.0f}"
                "<br><b>Click to inspect the nearest tower cells</b><extra></extra>"
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
            customdata=np.array(list(zip(
                np.repeat("recommendation", len(recs)),
                recs["rank"].astype(int),
                recs.candidate_id.astype(int),
                recs.nearest_tower_id.astype(int),
                recs.adm3_name.astype(str),
                recs.adm4_name.fillna("Unnamed").astype(str),
                recs.nearest_tower_km.astype(float),
                recs.population_2020.astype(float),
                recs.suitability_score.astype(float),
                recs.elevation_m.astype(float),
                recs.elevation_score.astype(float),
            )), dtype=object),
            hovertemplate=(
                "<b>Recommended site #%{customdata[1]}</b><br>%{customdata[5]}, %{customdata[4]}"
                "<br>Latitude: %{lat:.6f}<br>Longitude: %{lon:.6f}"
                "<br>Current gap: %{customdata[6]:.2f} km"
                "<br>Admin-4 population: %{customdata[7]:,.0f}"
                "<br>Elevation: %{customdata[9]:.0f} m"
                "<br>Elevation score: %{customdata[10]:.2f}"
                "<br>Suitability: %{customdata[8]:.1f}/100"
                "<br><b>Click to inspect this recommendation</b><extra></extra>"
            ),
            name="Recommended new sites",
        )
    )
    return fig


def _mapping_get(obj, key, default=None):
    """Read normal dicts and Streamlit's dictionary-like Plotly event objects."""
    if obj is None:
        return default
    try:
        return obj[key]
    except Exception:
        pass
    try:
        return getattr(obj, key)
    except Exception:
        pass
    try:
        return obj.get(key, default)
    except Exception:
        return default


def event_points(event):
    """Return Plotly selected points robustly across Streamlit event object versions."""
    if event is None:
        return []
    selection = _mapping_get(event, "selection", None)
    points = _mapping_get(selection, "points", []) if selection is not None else []
    return list(points or [])


def selected_tagged_point(event, tag):
    for point in reversed(event_points(event)):
        cd = _mapping_get(point, "customdata", []) or []
        try:
            cd = list(cd)
        except Exception:
            continue
        if cd and str(cd[0]) == tag:
            return cd
    return None


def _ai_report_text(a):
    if not a.get("ok"):
        return f"AI site assessment unavailable: {a.get('reason', 'Unknown reason')}"
    lines = [
        "XGBOOST TELECOM TOWER SITE ASSESSMENT",
        "",
        f"Coordinate: {a['lat']:.6f}, {a['lon']:.6f}",
        f"Admin-4: {a.get('adm4_name','')} — {a.get('adm3_name','')}",
        f"AI suitability probability: {a['score_100']:.2f}%",
        f"Decision threshold: {100*a['decision_threshold']:.1f}%",
        f"Decision: {a['decision']}",
        "",
        f"Population 2020 (Admin-4): {a.get('population_2020',0):,.0f}",
        f"Nearest observed tower-site: {a.get('nearest_tower_km',0):.3f} km",
        f"Elevation: {a.get('elevation_m') if a.get('elevation_m') is not None else 'N/A'} m",
        "",
        "Model inputs:",
    ]
    for feature, value in a.get("feature_values", {}).items():
        lines.append(f"- {feature}: {float(value):.4f}")
    lines.extend(["", "Main model drivers:"])
    for e in a.get("explanations", [])[:5]:
        lines.append(
            f"- {e['label']}: {e['direction']} (relative impact {e['relative_impact_pct']:.1f}%)"
        )
    lines.extend([
        "",
        "Important limitation:",
        a.get("model_warning", "Prototype model."),
        "Population, rural classification and safety are area-level proxies; tower gap and elevation are evaluated at the selected coordinate.",
        "This is a planning-screening result, not final RF, structural, land, permitting, power or backhaul approval.",
    ])
    return "\n".join(lines)


def render_ai_assessment(a, key_prefix="ai"):
    st.markdown("#### 🤖 XGBoost AI site assessment")
    if not a.get("ok"):
        st.warning(a.get("reason", "The selected coordinate cannot be evaluated."))
        return

    score = float(a["score_100"])
    threshold_pct = 100.0 * float(a["decision_threshold"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI suitability", f"{score:.1f}%")
    c2.metric("AI decision", a["decision"])
    c3.metric("Nearest tower", f"{float(a['nearest_tower_km']):.2f} km")
    c4.metric("Elevation", f"{float(a['elevation_m']):.0f} m" if a.get("elevation_m") is not None else "N/A")

    if a.get("is_optimal"):
        st.success(
            f"The trained XGBoost model classifies this point as an **optimal candidate**: "
            f"{score:.1f}% ≥ the provisional {threshold_pct:.1f}% decision threshold."
        )
    else:
        st.warning(
            f"The trained XGBoost model does **not** classify this point as optimal: "
            f"{score:.1f}% < the provisional {threshold_pct:.1f}% decision threshold."
        )

    st.write(
        f"**Selected point:** `{float(a['lat']):.6f}, {float(a['lon']):.6f}`  "
        f"\n**Administrative area:** {a.get('adm4_name','')} — {a.get('adm3_name','')}  "
        f"\n**Admin-4 population 2020:** {float(a.get('population_2020',0)):,.0f}  "
        f"\n**Nearest observed tower-site:** ID {a.get('nearest_tower_id','')} at "
        f"`{float(a.get('nearest_tower_lat',0)):.6f}, {float(a.get('nearest_tower_lon',0)):.6f}` "
        f"({a.get('nearest_tower_networks','')} | {a.get('nearest_tower_radios','')})"
    )

    feature_labels = {
        "gap_score": "Coverage gap",
        "population_score": "Population demand",
        "is_rural": "Rural priority",
        "safety_score": "Hazard safety",
        "elevation_score": "Elevation advantage",
    }
    rows = []
    for feature, value in a.get("feature_values", {}).items():
        rows.append({
            "Model feature": feature_labels.get(feature, feature),
            "Normalized value (0–1)": round(float(value), 4),
            "How it was obtained": a.get("feature_sources", {}).get(feature, ""),
        })
    st.markdown("**Inputs sent to XGBoost**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    expl = pd.DataFrame(a.get("explanations", []))
    if not expl.empty:
        expl = expl[["label", "direction", "relative_impact_pct", "value"]].copy()
        expl.columns = ["Factor", "Effect on prediction", "Relative model impact %", "Input value"]
        expl["Relative model impact %"] = expl["Relative model impact %"].round(1)
        expl["Input value"] = expl["Input value"].round(4)
        st.markdown("**Why the model decided this**")
        st.dataframe(expl, use_container_width=True, hide_index=True)
        st.caption(
            "Direction is based on XGBoost margin contributions. Relative impact is the share of absolute contribution magnitude for this prediction; it is not a causal percentage."
        )

    report = _ai_report_text(a)
    st.download_button(
        "⬇️ Download AI site assessment report",
        report.encode("utf-8"),
        file_name=f"xgboost_site_assessment_{float(a['lat']):.5f}_{float(a['lon']):.5f}.txt",
        mime="text/plain",
        key=f"{key_prefix}_download_ai_report",
    )
    st.caption(
        "Prototype limitation: the model was trained on pseudo-labels from the previous planning rule. "
        "Population/rural/safety are Admin-4-level proxies; tower gap and elevation are evaluated at the selected coordinate. "
        "Use this for planning screening, not final RF/site engineering approval."
    )


def build_ai_click_map(selected_areas, selected_point=None):
    if not HAS_CLICK_MAP:
        return None
    towns = expand_areas(selected_areas)
    boundary = filter_geojson(admin3_geo, "adm3_name", towns)
    source = candidates_all[candidates_all.adm3_name.isin(towns)]
    center = map_center(source)
    zoom = 9 if len(selected_areas) == 1 else 8
    m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    if boundary.get("features"):
        folium.GeoJson(
            boundary,
            name="Analysis boundary",
            style_function=lambda _: {
                "color": "#2457C5", "weight": 2, "fillColor": "#4C78FF", "fillOpacity": 0.06,
            },
        ).add_to(m)
    if selected_point:
        lat, lon = selected_point
        folium.Marker([lat, lon], tooltip="Selected AI assessment point").add_to(m)
    folium.LatLngPopup().add_to(m)
    return m


def show_nearest_tower_selection(event, candidate_source, key_prefix):
    cd = selected_tagged_point(event, "gap")
    if not cd or len(cd) < 3:
        st.caption("Click an orange/red gap point to inspect its nearest observed tower and the individual cell records grouped at that site.")
        return False
    try:
        candidate_id = int(float(cd[1]))
        tower_id = int(float(cd[2]))
    except Exception:
        st.warning("The selected gap point could not be interpreted.")
        return True

    crow = candidate_source[candidate_source.candidate_id.astype(int) == candidate_id]
    if crow.empty:
        crow = candidates_all[candidates_all.candidate_id.astype(int) == candidate_id]
    tw = tower_sites_lookup_all[tower_sites_lookup_all.yangon_tower_id == tower_id]
    if crow.empty or tw.empty:
        st.warning("The selected point could not be linked to its nearest observed tower-site proxy.")
        return True

    c = crow.iloc[0]
    t = tw.iloc[0]
    cells = tower_cells_lookup_all[tower_cells_lookup_all.yangon_tower_id == tower_id].copy()

    st.markdown("#### Selected gap → nearest tower cells")
    a, b, c3, d = st.columns(4)
    a.metric("Gap distance", f"{float(c['nearest_tower_km']):.2f} km")
    b.metric("Nearest tower-site ID", f"{tower_id}")
    c3.metric("Cell records at site", f"{len(cells):,}")
    d.metric("Admin-4 population", f"{float(c['population_2020']):,.0f}")
    st.write(
        f"**Gap point:** {c.get('adm4_name', 'Unnamed')} — {c.get('adm3_name', '')}  "
        f"\n**Gap coordinates:** `{float(c['lat']):.6f}, {float(c['lon']):.6f}`  "
        f"\n**Nearest tower:** {t.get('adm3_name', '')} at `{float(t['lat']):.6f}, {float(t['lon']):.6f}`  "
        f"\n**Networks:** {t.get('networks', '')} | **Radios:** {t.get('radios', '')}"
    )

    detail = go.Figure()
    detail.add_trace(go.Scattermap(
        lat=[float(c['lat']), float(t['lat'])], lon=[float(c['lon']), float(t['lon'])],
        mode="lines", line={"width": 3, "color": "#5b6470"}, hoverinfo="skip", name="Gap to nearest tower"
    ))
    detail.add_trace(go.Scattermap(
        lat=[float(c['lat'])], lon=[float(c['lon'])], mode="markers+text", text=["Gap point"], textposition="top center",
        marker={"size": 16, "color": "#ef7d00"},
        hovertemplate=f"<b>{c.get('adm4_name','Gap point')}</b><br>{float(c['lat']):.6f}, {float(c['lon']):.6f}<extra></extra>",
        name="Selected gap point"
    ))
    detail.add_trace(go.Scattermap(
        lat=[float(t['lat'])], lon=[float(t['lon'])], mode="markers+text", text=[f"Tower {tower_id}"], textposition="top center",
        marker={"size": 17, "color": "#3157d5"},
        hovertemplate=f"<b>Tower-site {tower_id}</b><br>{float(t['lat']):.6f}, {float(t['lon']):.6f}<br>{t.get('networks','')} | {t.get('radios','')}<extra></extra>",
        name="Nearest observed tower"
    ))
    mid_lat = (float(c['lat']) + float(t['lat'])) / 2
    mid_lon = (float(c['lon']) + float(t['lon'])) / 2
    dist = max(float(c['nearest_tower_km']), 0.2)
    zoom = float(np.clip(11.5 - np.log2(dist + 0.5), 5.0, 12.5))
    detail.update_layout(
        map={"style": "open-street-map", "center": {"lat": mid_lat, "lon": mid_lon}, "zoom": zoom},
        height=420,
        margin={"l": 8, "r": 8, "t": 10, "b": 8},
        legend={
            "orientation": "h", "y": 0.015, "yanchor": "bottom",
            "x": 0.5, "xanchor": "center",
            "bgcolor": "rgba(20,24,32,0.76)",
            "bordercolor": "rgba(255,255,255,0.20)", "borderwidth": 1,
            "font": {"color": "white", "size": 11},
        },
    )
    st.plotly_chart(detail, use_container_width=True, key=f"{key_prefix}_nearest_detail", config=MAP_PLOTLY_CONFIG)

    if len(cells):
        st.markdown("**Individual cell records grouped at this nearest tower-site proxy**")
        cell_cols = [x for x in ["radio", "Network", "MCC", "MNC", "TAC", "CID", "RANGE", "LAT", "LON"] if x in cells.columns]
        shown = cells[cell_cols].copy().rename(columns={"LAT": "cell_latitude", "LON": "cell_longitude", "RANGE": "reported_range_m"})
        st.dataframe(shown, use_container_width=True, hide_index=True, height=min(360, 70 + 35 * len(shown)))

    render_ai_assessment(assess_site(float(c['lat']), float(c['lon'])), key_prefix=f"{key_prefix}_gap_ai")
    return True


def show_recommendation_selection(event, recs, key_prefix):
    cd = selected_tagged_point(event, "recommendation")
    if not cd or len(cd) < 4:
        return False
    try:
        rank = int(float(cd[1]))
        candidate_id = int(float(cd[2]))
        tower_id = int(float(cd[3]))
    except Exception:
        st.warning("The selected recommendation could not be interpreted.")
        return True

    rr = recs[(recs["rank"].astype(int) == rank) & (recs.candidate_id.astype(int) == candidate_id)]
    if rr.empty:
        rr = recs[recs.candidate_id.astype(int) == candidate_id]
    if rr.empty:
        st.warning("The selected recommendation could not be linked to its recommendation row.")
        return True
    r = rr.iloc[0]
    tw = tower_sites_lookup_all[tower_sites_lookup_all.yangon_tower_id == tower_id]

    st.markdown("#### Selected recommended site")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommendation", f"#{int(r['rank'])}")
    c2.metric("Suitability", f"{float(r['suitability_score']):.2f}/100")
    c3.metric("Current tower gap", f"{float(r['nearest_tower_km']):.2f} km")
    c4.metric("Population 2020", f"{float(r['population_2020']):,.0f}")
    c5.metric("Elevation", f"{float(r.get('elevation_m', 0)):.0f} m")

    detail_cols = [
        "rank", "adm3_name", "adm4_name", "lat", "lon", "population_2020",
        "nearest_tower_id", "nearest_tower_lat", "nearest_tower_lon", "nearest_tower_km",
        "gap_score", "population_score", "is_rural", "elevation_m", "elevation_score",
        "earthquake_score", "cyclone_score", "hazard_score", "safety_score", "suitability_score",
    ]
    detail_cols = [x for x in detail_cols if x in rr.columns]
    detail = rr.iloc[[0]][detail_cols].copy().rename(columns={"lat": "recommended_latitude", "lon": "recommended_longitude"})
    for col in ["recommended_latitude", "recommended_longitude", "nearest_tower_lat", "nearest_tower_lon"]:
        if col in detail:
            detail[col] = detail[col].astype(float).round(6)
    if "nearest_tower_km" in detail:
        detail["nearest_tower_km"] = detail["nearest_tower_km"].astype(float).round(3)
    if "suitability_score" in detail:
        detail["suitability_score"] = detail["suitability_score"].astype(float).round(2)
    st.dataframe(detail, use_container_width=True, hide_index=True)
    st.write(
        f"**Recommended coordinates:** `{float(r['lat']):.6f}, {float(r['lon']):.6f}`  "
        f"\n**Place:** {r.get('adm4_name','Unnamed')} — {r.get('adm3_name','')}"
    )

    if not tw.empty:
        t = tw.iloc[0]
        focus = go.Figure()
        focus.add_trace(go.Scattermap(
            lat=[float(r['lat']), float(t['lat'])], lon=[float(r['lon']), float(t['lon'])],
            mode="lines", line={"width": 3, "color": "#5b6470"}, hoverinfo="skip", name="Current gap"
        ))
        focus.add_trace(go.Scattermap(
            lat=[float(r['lat'])], lon=[float(r['lon'])], mode="markers+text", text=[f"Recommendation #{int(r['rank'])}"], textposition="top center",
            marker={"size": 18, "color": "#00A878"},
            hovertemplate=f"<b>Recommended site #{int(r['rank'])}</b><br>{float(r['lat']):.6f}, {float(r['lon']):.6f}<extra></extra>",
            name="Recommended site"
        ))
        focus.add_trace(go.Scattermap(
            lat=[float(t['lat'])], lon=[float(t['lon'])], mode="markers+text", text=[f"Tower {tower_id}"], textposition="top center",
            marker={"size": 17, "color": "#3157d5"},
            hovertemplate=f"<b>Nearest tower-site {tower_id}</b><br>{float(t['lat']):.6f}, {float(t['lon']):.6f}<br>{t.get('networks','')} | {t.get('radios','')}<extra></extra>",
            name="Nearest existing tower"
        ))
        mid_lat = (float(r['lat']) + float(t['lat'])) / 2
        mid_lon = (float(r['lon']) + float(t['lon'])) / 2
        dist = max(float(r['nearest_tower_km']), 0.2)
        zoom = float(np.clip(11.5 - np.log2(dist + 0.5), 5.0, 12.5))
        focus.update_layout(
            map={"style": "open-street-map", "center": {"lat": mid_lat, "lon": mid_lon}, "zoom": zoom},
            height=400,
            margin={"l": 8, "r": 8, "t": 10, "b": 8},
            legend={
                "orientation": "h", "y": 0.015, "yanchor": "bottom",
                "x": 0.5, "xanchor": "center",
                "bgcolor": "rgba(20,24,32,0.76)",
                "bordercolor": "rgba(255,255,255,0.20)", "borderwidth": 1,
                "font": {"color": "white", "size": 11},
            },
        )
        st.plotly_chart(focus, use_container_width=True, key=f"{key_prefix}_recommendation_detail", config=MAP_PLOTLY_CONFIG)

        cells = tower_cells_lookup_all[tower_cells_lookup_all.yangon_tower_id == tower_id].copy()
        if len(cells):
            st.markdown("**Cell records at the nearest existing tower-site proxy**")
            cell_cols = [x for x in ["radio", "Network", "MCC", "MNC", "TAC", "CID", "RANGE", "LAT", "LON"] if x in cells.columns]
            shown = cells[cell_cols].copy().rename(columns={"LAT": "cell_latitude", "LON": "cell_longitude", "RANGE": "reported_range_m"})
            st.dataframe(shown, use_container_width=True, hide_index=True, height=min(330, 70 + 35 * len(shown)))

    render_ai_assessment(assess_site(float(r['lat']), float(r['lon'])), key_prefix=f"{key_prefix}_recommend_ai")
    return True


def show_map_selection(event, candidate_source, recs, key_prefix):
    tags = []
    for point in event_points(event):
        cd = _mapping_get(point, "customdata", []) or []
        try:
            cd = list(cd)
        except Exception:
            cd = []
        if cd:
            tags.append(str(cd[0]))
    if tags and tags[-1] == "recommendation":
        show_recommendation_selection(event, recs, key_prefix)
        return
    if tags and tags[-1] == "gap":
        show_nearest_tower_selection(event, candidate_source, key_prefix)
        return
    st.caption("Click an orange/red gap point to inspect its nearest tower cells, or click a green recommendation marker to inspect the recommendation table.")


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
            colorbar={
                "title": {"text": "Flood freq.", "side": "right"},
                "x": 1.02, "xanchor": "left",
                "y": 0.47, "yanchor": "middle",
                "len": 0.68, "thickness": 16, "outlinewidth": 0,
            },
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
st.sidebar.title("Planning controls")
st.sidebar.caption("Set the study area and the three assumptions most useful for planning. Technical model controls are kept under Advanced settings.")

study_area = st.sidebar.selectbox(
    "Study area",
    ["All project areas", "Yangon City", "Hmawbi", "Thanlyin", "Kyauktan", "Custom selection"],
    index=0,
    help="Choose one focus area, all four project areas, or Custom selection for a combination.",
)
if study_area == "All project areas":
    selected_areas = ANALYSIS_AREAS
elif study_area == "Custom selection":
    selected_areas = st.sidebar.multiselect(
        "Choose areas",
        options=ANALYSIS_AREAS,
        default=ANALYSIS_AREAS,
        help="Select any combination of the four project areas.",
    )
    if not selected_areas:
        selected_areas = ANALYSIS_AREAS
else:
    selected_areas = [study_area]

selected_townships = expand_areas(selected_areas)

st.sidebar.markdown("#### Coverage planning")
service_radius_km = st.sidebar.slider(
    "Population service radius",
    1.0, 15.0, 5.0, 0.5,
    format="%.1f km",
    help="Population within this distance of an observed tower-site proxy is counted as geographically covered. This is a planning radius, not an RF propagation model.",
)
threshold_km = st.sidebar.slider(
    "Underserved threshold",
    2.0, 20.0, 5.0, 0.5,
    format="%.1f km",
    help="Admin-4 areas whose representative point is at least this far from its nearest observed tower-site proxy are flagged as underserved.",
)
n_sites = st.sidebar.slider(
    "New tower candidates",
    3, 20, 10,
    help="How many high-priority candidate areas the GeoAI recommendation module should return.",
)
recommendation_engine = st.sidebar.selectbox(
    "Recommendation engine",
    ["XGBoost AI (trained model)", "Rule-based baseline"],
    index=0,
    help="XGBoost uses the saved trained model. Rule-based baseline uses the original weighted planning formula.",
)

with st.sidebar.expander("Advanced settings", expanded=False):
    st.caption("Optional. Keep the defaults for a simple presentation.")
    min_spacing = st.slider(
        "Minimum spacing between new sites",
        2.0, 20.0, 8.0, 1.0,
        format="%.0f km",
        help="Prevents recommended sites from clustering too close together.",
    )
    if recommendation_engine.startswith("XGBoost"):
        st.markdown("**XGBoost model**")
        st.caption("The trained model uses fixed learned tree parameters. No manual scoring weights are applied in XGBoost mode.")
    else:
        st.markdown("**Rule-based scoring weights**")
        w_gap = st.slider("Coverage need", 0.0, 1.0, 0.40, 0.05)
        w_pop = st.slider("Population demand", 0.0, 1.0, 0.25, 0.05)
        w_rural = st.slider("Rural priority", 0.0, 1.0, 0.10, 0.05)
        w_safe = st.slider("Hazard safety", 0.0, 1.0, 0.10, 0.05)
        w_elev = st.slider("Elevation advantage", 0.0, 1.0, 0.15, 0.05, help="Higher sampled terrain elevation receives a higher suitability score. Elevation is normalized across the current Yangon candidate locations.")

st.sidebar.caption(
    f"Active: {', '.join(selected_areas)} • service radius {service_radius_km:.1f} km • "
    f"underserved ≥ {threshold_km:.1f} km • {n_sites} candidate sites • "
    f"engine: {'XGBoost AI' if recommendation_engine.startswith('XGBoost') else 'rule-based'}"
)

candidates = candidates_all[candidates_all.adm3_name.isin(selected_townships)].copy()
towers = tower_sites_all[tower_sites_all.adm3_name.isin(selected_townships)].copy()
summary = coverage_summary(candidates, threshold_km)
summary_area = area_summary(candidates, selected_areas, threshold_km)
if recommendation_engine.startswith("XGBoost"):
    recs = recommend_sites_xgb(candidates, n_sites, min_spacing)
else:
    recs = recommend_sites(candidates, n_sites, min_spacing, w_gap, w_pop, w_rural, w_safe, w_elev)
    recs["recommendation_engine"] = "Rule-based baseline"

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
t_overview, t_population, t_gap, t_recommend, t_ai, t_disaster, t_rain, t_method = st.tabs(
    [
        "Overview",
        "Population & tower load",
        "Underserved areas",
        "Tower recommendations",
        "🤖 AI Site Checker",
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
    st.caption("Tip: click an orange/red Admin-4 gap point to inspect its nearest tower and cells. Green numbered recommendation markers are also clickable.")
    overview_event = st.plotly_chart(fig, use_container_width=True, key="overview_gap_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
    overview_detail = st.container()
    with overview_detail:
        show_map_selection(overview_event, candidates, recs, "overview")

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
                    "colorbar": {
                        "title": {"text": "Population load", "side": "right"},
                        "x": 1.02, "xanchor": "left",
                        "y": 0.47, "yanchor": "middle",
                        "len": 0.68, "thickness": 16, "outlinewidth": 0,
                    },
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
        st.plotly_chart(fig, use_container_width=True, config=MAP_PLOTLY_CONFIG)

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
        gap_source = underserved if len(underserved) else candidates
        add_candidates(fig, gap_source, "Underserved admin-4 units")
        st.caption("Click an orange/red gap point to display the exact nearest observed tower-site proxy and its underlying cell records.")
        under_event = st.plotly_chart(fig, use_container_width=True, key="underserved_gap_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
        under_detail = st.container()
        with under_detail:
            show_nearest_tower_selection(under_event, gap_source, "underserved")
    with c2:
        st.metric("Population in underserved Admin-4 units", f"{underserved.population_2020.sum():,.0f}")
        top = underserved[[
            "analysis_area", "adm3_name", "adm4_name", "population_2020", "nearest_tower_km",
            "nearest_tower_id", "nearest_tower_lat", "nearest_tower_lon", "hazard_score"
        ]].head(25).copy()
        if len(top):
            top["population_2020"] = top.population_2020.round(0).astype(int)
            top["nearest_tower_km"] = top.nearest_tower_km.round(2)
            top["nearest_tower_lat"] = top.nearest_tower_lat.round(6)
            top["nearest_tower_lon"] = top.nearest_tower_lon.round(6)
            top["hazard_score"] = (100 * top.hazard_score).round(0).astype(int)
        top.columns = ["Area", "Township", "Admin-4", "Population 2020", "Nearest site km", "Tower ID", "Tower latitude", "Tower longitude", "Hazard /100"]
        st.dataframe(top, use_container_width=True, hide_index=True, height=490)

    st.info(
        "This improves the old 'distance only' view: a 10 km gap with 20,000 people can now be distinguished from a 10 km gap with 500 people."
    )

with t_recommend:
    st.subheader("Population-aware GeoAI tower recommendations")
    if recommendation_engine.startswith("XGBoost"):
        st.write(
            "Candidate areas are ranked by the trained XGBoost model using coverage gap, population demand, rural priority, hazard safety and terrain elevation. "
            "The model probability becomes the 0–100 suitability score, then the minimum-spacing rule prevents recommendations from clustering together."
        )
        st.caption("Recommendation engine: trained `models/tower_site_xgb.json` XGBClassifier.")
    else:
        st.write(
            "Candidate areas are ranked using the original weighted rule: coverage gap, population demand, rural priority, hazard safety and terrain elevation. "
            "Higher sampled elevation receives more suitability priority, then a minimum-spacing rule prevents recommendations from clustering together."
        )
        st.caption("Recommendation engine: rule-based baseline. Use the sidebar selector to switch to XGBoost AI.")

    fig = base_map(selected_areas, recs, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    add_towers(fig, towers, max_points=1600)
    add_recommendations(fig, recs)
    st.caption("Click a green numbered recommendation marker to display its complete recommendation row, exact coordinates, nearest existing tower, and tower cell records.")
    rec_event = st.plotly_chart(fig, use_container_width=True, key="recommendation_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
    rec_detail = st.container()
    with rec_detail:
        if not show_recommendation_selection(rec_event, recs, "recommendation"):
            st.caption("No recommendation selected yet. Click a green numbered point on the map.")

    recs_display = add_area_label(recs)
    export_cols = [
        "rank", "analysis_area", "adm3_name", "adm4_name", "lat", "lon", "population_2020",
        "nearest_tower_id", "nearest_tower_lat", "nearest_tower_lon", "nearest_tower_km",
        "gap_score", "population_score", "elevation_m", "elevation_score", "hazard_score", "safety_score", "suitability_score",
    ]
    for optional_col in ["ai_probability", "ai_decision", "recommendation_engine"]:
        if optional_col in recs_display.columns:
            export_cols.append(optional_col)
    table = recs_display[export_cols].copy()
    table = table.rename(columns={"lat": "latitude", "lon": "longitude"})
    if "ai_probability" in table:
        table["ai_probability"] = (100 * table["ai_probability"].astype(float)).round(2)
        table = table.rename(columns={"ai_probability": "xgboost_probability_pct"})
    for c in ["latitude", "longitude", "nearest_tower_lat", "nearest_tower_lon"]:
        table[c] = table[c].astype(float).round(6)
    for c in ["nearest_tower_km", "gap_score", "population_score", "elevation_score", "hazard_score", "safety_score"]:
        table[c] = table[c].astype(float).round(4)
    if "elevation_m" in table:
        table["elevation_m"] = table["elevation_m"].astype(float).round(1)
    table["suitability_score"] = table.suitability_score.astype(float).round(2)
    table["population_2020"] = table.population_2020.round(0).astype(int)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download recommended places with latitude/longitude (CSV)",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="yangon_geoai_recommended_places_coordinates.csv",
        mime="text/csv",
    )
    st.caption("Latitude/longitude are candidate-zone coordinates for field and engineering investigation, not final approved construction coordinates.")

with t_ai:
    st.subheader("🤖 XGBoost AI Tower Site Checker")
    status = model_status()
    st.write(
        "Click a coordinate inside the project boundary. The dashboard builds the same five features used during training, "
        "runs the saved `tower_site_xgb.json` model, and returns an optimal/not-optimal decision with a downloadable report."
    )
    st.info(
        f"Loaded trained XGBClassifier • {status['training_rows']} training rows • "
        f"{len(status['features'])} model features • provisional optimal threshold {100*status['threshold']:.0f}%. "
        "The legacy recommendation-weight sliders do not change this trained model."
    )

    previous = st.session_state.get("ai_site_checker_point")
    clicked_point = None

    if HAS_CLICK_MAP:
        m = build_ai_click_map(selected_areas, previous)
        st.caption("Click anywhere inside the blue analysis boundary. The latitude/longitude popup is also shown on the map.")
        click_state = st_folium(
            m,
            width=1100,
            height=560,
            returned_objects=["last_clicked"],
            key="xgb_ai_click_map",
        )
        if click_state and click_state.get("last_clicked"):
            raw = click_state["last_clicked"]
            clicked_point = (float(raw["lat"]), float(raw["lng"]))
            st.session_state["ai_site_checker_point"] = clicked_point
        elif previous:
            clicked_point = previous
    else:
        st.warning(
            "Interactive blank-map click support requires the `streamlit-folium` package. "
            "The coordinate checker below remains fully functional."
        )

    with st.expander("Enter an exact coordinate instead", expanded=not HAS_CLICK_MAP):
        default_lat = float(clicked_point[0] if clicked_point else 16.86)
        default_lon = float(clicked_point[1] if clicked_point else 96.20)
        x1, x2, x3 = st.columns([1, 1, 0.7])
        with x1:
            manual_lat = st.number_input("Latitude", value=default_lat, format="%.6f", key="ai_manual_lat")
        with x2:
            manual_lon = st.number_input("Longitude", value=default_lon, format="%.6f", key="ai_manual_lon")
        with x3:
            st.write("")
            st.write("")
            if st.button("Assess coordinate", type="primary", key="ai_manual_assess"):
                clicked_point = (float(manual_lat), float(manual_lon))
                st.session_state["ai_site_checker_point"] = clicked_point

    if clicked_point:
        render_ai_assessment(assess_site(clicked_point[0], clicked_point[1]), key_prefix="ai_checker")
    else:
        st.caption("No point selected yet. Click the map or enter a coordinate to run the trained XGBoost model.")


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
                "colorbar": {
                    "title": {"text": "Scenario risk", "side": "right"},
                    "x": 1.02, "xanchor": "left",
                    "y": 0.47, "yanchor": "middle",
                    "len": 0.68, "thickness": 16, "outlinewidth": 0,
                },
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
    st.plotly_chart(fig, use_container_width=True, config=MAP_PLOTLY_CONFIG)

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
    st.plotly_chart(fig, use_container_width=True, config=MAP_PLOTLY_CONFIG)

with t_method:
    st.subheader("Data inventory and methodology")
    st.markdown(
        "**XGBoost site model:** the deployed `models/tower_site_xgb.json` classifier uses coverage gap, population demand, rural priority, hazard safety and elevation advantage. "
        "For arbitrary map clicks, tower gap and elevation are evaluated at the exact coordinate; population, rural classification and hazard safety come from the containing Admin-4 area. "
        "The current training target is a pseudo-label from the prior planning rule, so the model is an MVP screening model rather than operator-validated deployment intelligence."
    )
    drows = pd.DataFrame([
        ["WorldPop population GeoTIFF", meta.get("population_grid_points", 0), "Usable", "Population count per raster pixel; reference year 2020"],
        ["Yangon elevation GeoTIFF", meta.get("elevation_candidate_count", 0), "Usable", f"Elevation sampled at candidate points; {meta.get('elevation_candidate_min_m', 0):.0f}–{meta.get('elevation_candidate_max_m', 0):.0f} m in current candidate set"],
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
        - Elevation score: `(candidate elevation - minimum candidate elevation) / (maximum - minimum)`. Higher terrain receives a higher score.
        - Default suitability weights: 40% gap + 25% population + 10% rural + 10% safety + 15% elevation.

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
