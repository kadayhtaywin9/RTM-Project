from __future__ import annotations

import json
import base64
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from site_ai import assess_site, model_status, recommend_sites_xgb
from hazard_ai import hazard_model_status, run_hazard_ai

try:
    import folium
    from streamlit_folium import st_folium
    HAS_CLICK_MAP = True
except Exception:
    folium = None
    st_folium = None
    HAS_CLICK_MAP = False

from geoai_engine import (
    coverage_summary,
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

st.set_page_config(page_title="Yangon Telecom Planning & Resilience", page_icon="📡", layout="wide")

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


def image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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


def _level(value, low=0.34, high=0.67):
    """Convert a normalized 0-1 planning score into plain-language bands."""
    try:
        value = float(value)
    except Exception:
        return "Unknown"
    if value >= high:
        return "High"
    if value >= low:
        return "Moderate"
    return "Low"


def _recommendation_label(row):
    """Use decision-support language instead of implying final engineering approval."""
    decision = str(row.get("ai_decision", "")).upper()
    if decision == "OPTIMAL CANDIDATE":
        return "Recommended for field review"
    try:
        score = float(row.get("suitability_score", 0))
    except Exception:
        score = 0.0
    if score >= 70:
        return "Recommended for field review"
    if score >= 50:
        return "Consider for further review"
    return "Lower priority"


def _recommendation_reason(row):
    """Explain a recommendation using the planning factors already present in the model."""
    reasons = []
    gap_km = float(row.get("nearest_tower_km", 0) or 0)
    pop_score = float(row.get("population_score", 0) or 0)
    hazard = float(row.get("hazard_score", 0) or 0)
    elevation = float(row.get("elevation_score", 0) or 0)

    if gap_km >= 8:
        reasons.append(f"it is {gap_km:.1f} km from the nearest mapped tower site")
    elif gap_km >= 5:
        reasons.append(f"it has a noticeable {gap_km:.1f} km planning gap to the nearest mapped tower site")
    else:
        reasons.append(f"it is {gap_km:.1f} km from the nearest mapped tower site")

    if pop_score >= 0.67:
        reasons.append("the surrounding population need is high")
    elif pop_score >= 0.34:
        reasons.append("the surrounding population need is moderate")

    if hazard <= 0.33:
        reasons.append("disaster exposure is comparatively low")
    elif hazard >= 0.67:
        reasons.append("disaster exposure is high, so resilient design would be important")
    else:
        reasons.append("disaster exposure is moderate")

    if elevation >= 0.67:
        reasons.append("the terrain score is relatively favorable")

    if len(reasons) == 1:
        return reasons[0].capitalize() + "."
    return reasons[0].capitalize() + ", " + ", and ".join(reasons[1:]) + "."


def _assessment_reason(a):
    features = a.get("feature_values", {}) if isinstance(a, dict) else {}
    row = {
        "nearest_tower_km": a.get("nearest_tower_km", 0),
        "population_score": features.get("population_score", 0),
        "hazard_score": 1.0 - float(features.get("safety_score", 0) or 0),
        "elevation_score": features.get("elevation_score", 0),
    }
    return _recommendation_reason(row)


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
                name="Study area boundary",
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
                "<b>Existing telecom tower</b><br>Township: %{customdata[0]}"
                "<br>Technology: %{customdata[1]}<br>Mapped cell records: %{customdata[2]}"
                "<br>Estimated nearby population: %{customdata[3]:,.0f}<extra></extra>"
            ),
            name="Existing telecom towers",
        )
    )
    return fig


def add_candidates(fig, df, name="Ward / Village Tract areas"):
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
                "<b>%{customdata[4]}</b><br>Township: %{customdata[3]}"
                "<br>Nearest existing tower: %{customdata[5]:.2f} km"
                "<br>Population 2020: %{customdata[7]:,.0f}"
                "<br><b>Click to see why this area is underserved</b><extra></extra>"
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
                "<b>Suggested tower location #%{customdata[1]}</b><br>%{customdata[5]} — %{customdata[4]}"
                "<br>Nearest existing tower: %{customdata[6]:.2f} km"
                "<br>Population 2020: %{customdata[7]:,.0f}"
                "<br>Overall site score: %{customdata[8]:.1f}/100"
                "<br><b>Click to see the recommendation</b><extra></extra>"
            ),
            name="Suggested tower locations",
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


def render_ai_assessment(a, key_prefix="ai", technical_only=False):
    if not technical_only:
        st.markdown("#### Site assessment")
    if not a.get("ok"):
        st.warning(a.get("reason", "The selected location cannot be evaluated."))
        return

    score = float(a["score_100"])
    if not technical_only:
        label = "Recommended for field review" if a.get("is_optimal") else "Lower priority"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall site score", f"{score:.1f}/100")
        c2.metric("Recommendation", label)
        c3.metric("Nearest existing tower", f"{float(a['nearest_tower_km']):.2f} km")
        c4.metric("Elevation", f"{float(a['elevation_m']):.0f} m" if a.get("elevation_m") is not None else "N/A")

        if a.get("is_optimal"):
            st.success("This location is a strong planning candidate and should be considered for field review.")
        else:
            st.warning("This location is a lower-priority planning candidate compared with stronger alternatives in the study area.")

        st.markdown(f"**Why this result:** {_assessment_reason(a)}")
        st.write(
            f"**Township:** {a.get('adm3_name','')}  "
            f"\n**Ward / Village Tract:** {a.get('adm4_name','')}  "
            f"\n**Population 2020:** {float(a.get('population_2020',0)):,.0f}  "
            f"\n**Selected coordinates:** `{float(a['lat']):.6f}, {float(a['lon']):.6f}`"
        )
        st.caption(
            "Planning recommendation only. A field survey, RF study, land/access check, power/backhaul review and regulatory approval are still required before construction."
        )

    details_context = nullcontext() if technical_only else st.expander("Technical model details", expanded=False)
    with details_context:
        threshold_pct = 100.0 * float(a["decision_threshold"])
        st.caption(
            f"Model: GeoVision AI • model score {score:.2f}% • decision threshold {threshold_pct:.1f}% • raw decision: {a['decision']}"
        )
        st.write(
            f"**Nearest tower ID:** {a.get('nearest_tower_id','')} at "
            f"`{float(a.get('nearest_tower_lat',0)):.6f}, {float(a.get('nearest_tower_lon',0)):.6f}` "
            f"({a.get('nearest_tower_networks','')} | {a.get('nearest_tower_radios','')})"
        )

        feature_labels = {
            "gap_score": "Coverage need",
            "population_score": "Population need",
            "is_rural": "Rural priority",
            "safety_score": "Disaster safety",
            "elevation_score": "Terrain suitability",
        }
        rows = []
        for feature, value in a.get("feature_values", {}).items():
            rows.append({
                "Model factor": feature_labels.get(feature, feature),
                "Normalized value (0–1)": round(float(value), 4),
                "Data source / calculation": a.get("feature_sources", {}).get(feature, ""),
            })
        st.markdown("**Inputs used by the model**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        expl = pd.DataFrame(a.get("explanations", []))
        if not expl.empty:
            expl = expl[["label", "direction", "relative_impact_pct", "value"]].copy()
            expl.columns = ["Factor", "Effect on model score", "Relative model impact %", "Input value"]
            expl["Relative model impact %"] = expl["Relative model impact %"].round(1)
            expl["Input value"] = expl["Input value"].round(4)
            st.markdown("**Model explanation**")
            st.dataframe(expl, use_container_width=True, hide_index=True)
            st.caption(
                "Relative model impact describes this model prediction only; it is not a causal percentage."
            )

        report = _ai_report_text(a)
        st.download_button(
            "⬇️ Download technical site assessment report",
            report.encode("utf-8"),
            file_name=f"xgboost_site_assessment_{float(a['lat']):.5f}_{float(a['lon']):.5f}.txt",
            mime="text/plain",
            key=f"{key_prefix}_download_ai_report",
        )
        st.caption(
            "Prototype limitation: the trained model uses planning proxies and pseudo-labels. Use it for screening and prioritization, not final engineering approval."
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
            name="Study area boundary",
            style_function=lambda _: {
                "color": "#2457C5", "weight": 2, "fillColor": "#4C78FF", "fillOpacity": 0.06,
            },
        ).add_to(m)
    if selected_point:
        lat, lon = selected_point
        folium.Marker([lat, lon], tooltip="Selected location").add_to(m)
    folium.LatLngPopup().add_to(m)
    return m


def show_nearest_tower_selection(event, candidate_source, key_prefix):
    cd = selected_tagged_point(event, "gap")
    if not cd or len(cd) < 3:
        st.caption("Select an orange/red area to see its nearest existing tower and the local population context.")
        return False
    try:
        candidate_id = int(float(cd[1]))
        tower_id = int(float(cd[2]))
    except Exception:
        st.warning("The selected underserved area could not be interpreted.")
        return True

    crow = candidate_source[candidate_source.candidate_id.astype(int) == candidate_id]
    if crow.empty:
        crow = candidates_all[candidates_all.candidate_id.astype(int) == candidate_id]
    tw = tower_sites_lookup_all[tower_sites_lookup_all.yangon_tower_id == tower_id]
    if crow.empty or tw.empty:
        st.warning("The selected area could not be linked to its nearest mapped tower site.")
        return True

    c = crow.iloc[0]
    t = tw.iloc[0]
    cells = tower_cells_lookup_all[tower_cells_lookup_all.yangon_tower_id == tower_id].copy()

    st.markdown("#### Why this area is underserved")
    a, b, c3, d = st.columns(4)
    a.metric("Distance to nearest tower", f"{float(c['nearest_tower_km']):.2f} km")
    b.metric("Nearest tower ID", f"{tower_id}")
    c3.metric("Mapped cell records", f"{len(cells):,}")
    d.metric("Local population 2020", f"{float(c['population_2020']):,.0f}")
    st.write(
        f"**Ward / Village Tract:** {c.get('adm4_name', 'Unnamed')}  "
        f"\n**Township:** {c.get('adm3_name', '')}  "
        f"\n**Area coordinates:** `{float(c['lat']):.6f}, {float(c['lon']):.6f}`  "
        f"\n**Nearest existing tower:** {t.get('adm3_name', '')} at `{float(t['lat']):.6f}, {float(t['lon']):.6f}`  "
        f"\n**Networks:** {t.get('networks', '')} | **Radios:** {t.get('radios', '')}"
    )

    detail = go.Figure()
    detail.add_trace(go.Scattermap(
        lat=[float(c['lat']), float(t['lat'])], lon=[float(c['lon']), float(t['lon'])],
        mode="lines", line={"width": 3, "color": "#5b6470"}, hoverinfo="skip", name="Area to nearest tower"
    ))
    detail.add_trace(go.Scattermap(
        lat=[float(c['lat'])], lon=[float(c['lon'])], mode="markers+text", text=["Selected area"], textposition="top center",
        marker={"size": 16, "color": "#ef7d00"},
        hovertemplate=f"<b>{c.get('adm4_name','Selected area')}</b><br>{float(c['lat']):.6f}, {float(c['lon']):.6f}<extra></extra>",
        name="Selected area"
    ))
    detail.add_trace(go.Scattermap(
        lat=[float(t['lat'])], lon=[float(t['lon'])], mode="markers+text", text=[f"Tower {tower_id}"], textposition="top center",
        marker={"size": 17, "color": "#3157d5"},
        hovertemplate=f"<b>Existing tower {tower_id}</b><br>{float(t['lat']):.6f}, {float(t['lon']):.6f}<br>{t.get('networks','')} | {t.get('radios','')}<extra></extra>",
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
        st.markdown("**Technical cell records at the nearest mapped tower**")
        cell_cols = [x for x in ["radio", "Network", "MCC", "MNC", "TAC", "CID", "RANGE", "LAT", "LON"] if x in cells.columns]
        shown = cells[cell_cols].copy().rename(columns={"LAT": "cell_latitude", "LON": "cell_longitude", "RANGE": "reported_range_m"})
        st.dataframe(shown, use_container_width=True, hide_index=True, height=min(360, 70 + 35 * len(shown)))

    with st.expander("Model assessment for this gap point", expanded=False):
        render_ai_assessment(assess_site(float(c['lat']), float(c['lon'])), key_prefix=f"{key_prefix}_gap_ai", technical_only=True)
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

    st.markdown(f"#### Suggested tower location #{int(r['rank'])}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall site score", f"{float(r['suitability_score']):.1f}/100")
    c2.metric("Nearest existing tower", f"{float(r['nearest_tower_km']):.2f} km")
    c3.metric("Population 2020", f"{float(r['population_2020']):,.0f}")
    c4.metric("Disaster risk", _level(float(r.get('hazard_score', 0))))

    status_label = _recommendation_label(r)
    st.success(f"**Recommendation: {status_label}**")
    st.markdown(f"**Why this location:** {_recommendation_reason(r)}")
    st.write(
        f"**Township:** {r.get('adm3_name','')}  "
        f"\n**Ward / Village Tract:** {r.get('adm4_name','Unnamed')}  "
        f"\n**Suggested coordinates:** `{float(r['lat']):.6f}, {float(r['lon']):.6f}`"
    )
    st.caption(
        "This is a planning recommendation. Confirm the site with RF, land/access, structural, power, backhaul and regulatory checks before construction."
    )

    detail_cols = [
        "rank", "adm3_name", "adm4_name", "lat", "lon", "population_2020",
        "nearest_tower_id", "nearest_tower_lat", "nearest_tower_lon", "nearest_tower_km",
        "gap_score", "population_score", "is_rural", "elevation_m", "elevation_score",
        "earthquake_score", "cyclone_score", "hazard_score", "safety_score", "suitability_score",
        "ai_probability", "ai_decision", "recommendation_engine",
    ]
    detail_cols = [x for x in detail_cols if x in rr.columns]
    detail = rr.iloc[[0]][detail_cols].copy().rename(columns={
        "rank": "Rank",
        "adm3_name": "Township",
        "adm4_name": "Ward / Village Tract",
        "lat": "Suggested Latitude",
        "lon": "Suggested Longitude",
        "population_2020": "Population 2020",
        "nearest_tower_id": "Nearest Tower ID",
        "nearest_tower_lat": "Nearest Tower Latitude",
        "nearest_tower_lon": "Nearest Tower Longitude",
        "nearest_tower_km": "Nearest Tower Distance (km)",
        "gap_score": "Coverage Need Score",
        "population_score": "Population Need Score",
        "is_rural": "Rural Priority",
        "elevation_m": "Elevation (m)",
        "elevation_score": "Terrain Suitability Score",
        "earthquake_score": "Earthquake Risk Score",
        "cyclone_score": "Cyclone Risk Score",
        "hazard_score": "Disaster Risk Score",
        "safety_score": "Disaster Safety Score",
        "suitability_score": "Overall Site Score",
        "ai_probability": "Model Recommendation Probability",
        "ai_decision": "Model Decision",
        "recommendation_engine": "Recommendation Method",
    })
    for col in ["Suggested Latitude", "Suggested Longitude", "Nearest Tower Latitude", "Nearest Tower Longitude"]:
        if col in detail:
            detail[col] = detail[col].astype(float).round(6)
    if "Nearest Tower Distance (km)" in detail:
        detail["Nearest Tower Distance (km)"] = detail["Nearest Tower Distance (km)"].astype(float).round(3)
    if "Overall Site Score" in detail:
        detail["Overall Site Score"] = detail["Overall Site Score"].astype(float).round(2)
    with st.expander("Technical recommendation data", expanded=False):
        st.dataframe(detail, use_container_width=True, hide_index=True)

    if not tw.empty:
        t = tw.iloc[0]
        focus = go.Figure()
        focus.add_trace(go.Scattermap(
            lat=[float(r['lat']), float(t['lat'])], lon=[float(r['lon']), float(t['lon'])],
            mode="lines", line={"width": 3, "color": "#5b6470"}, hoverinfo="skip", name="Current gap"
        ))
        focus.add_trace(go.Scattermap(
            lat=[float(r['lat'])], lon=[float(r['lon'])], mode="markers+text", text=[f"Suggested site #{int(r['rank'])}"], textposition="top center",
            marker={"size": 18, "color": "#00A878"},
            hovertemplate=f"<b>Suggested tower location #{int(r['rank'])}</b><br>{float(r['lat']):.6f}, {float(r['lon']):.6f}<extra></extra>",
            name="Suggested location"
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
            st.markdown("**Technical cell records at the nearest existing tower**")
            cell_cols = [x for x in ["radio", "Network", "MCC", "MNC", "TAC", "CID", "RANGE", "LAT", "LON"] if x in cells.columns]
            shown = cells[cell_cols].copy().rename(columns={"LAT": "cell_latitude", "LON": "cell_longitude", "RANGE": "reported_range_m"})
            st.dataframe(shown, use_container_width=True, hide_index=True, height=min(330, 70 + 35 * len(shown)))

    with st.expander("Model explanation for this recommendation", expanded=False):
        render_ai_assessment(assess_site(float(r['lat']), float(r['lon'])), key_prefix=f"{key_prefix}_recommend_ai", technical_only=True)
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
    st.caption("Click an orange/red underserved area to see its nearest tower, or click a green suggested location to see why it is recommended.")


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
            "Local areas": int(len(subset)),
            "Population 2020": int(round(subset.population_2020.sum())),
            "Median tower distance (km)": round(float(subset.nearest_tower_km.median()), 2),
            "Largest tower distance (km)": round(float(subset.nearest_tower_km.max()), 2),
            "Underserved local areas": int(underserved.sum()),
            "Population in underserved areas": int(round(subset.loc[underserved, "population_2020"].sum())),
        })
    return pd.DataFrame(rows)


# ---------- sidebar ----------
st.sidebar.title("Planning settings")
st.sidebar.caption("Choose the area and planning assumptions. Keep the defaults for a quick demo.")

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
    "Recommendation method",
    ["GeoVision AI (trained model)", "Rule-based baseline"],
    index=0,
    format_func=lambda x: "GeoVision AI (recommended)" if x.startswith("GeoVision AI") else "Planning rules (comparison)",
    help="GeoVision AI is the default recommendation method. The planning-rule option is kept for comparison and validation.",
)

with st.sidebar.expander("Advanced settings", expanded=False):
    st.caption("Optional. Keep the defaults for a simple presentation.")
    min_spacing = st.slider(
        "Minimum spacing between new sites",
        2.0, 20.0, 8.0, 1.0,
        format="%.0f km",
        help="Prevents recommended sites from clustering too close together.",
    )
    if recommendation_engine.startswith("GeoVision AI"):
        st.markdown("**GeoVision AI**")
        st.caption("GeoVision AI uses fixed learned tree parameters. No manual scoring weights are applied in GeoVision AI mode.")
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
    f"method: {'GeoVision AI' if recommendation_engine.startswith('GeoVision AI') else 'planning rules'}"
)

candidates = candidates_all[candidates_all.adm3_name.isin(selected_townships)].copy()
towers = tower_sites_all[tower_sites_all.adm3_name.isin(selected_townships)].copy()
summary = coverage_summary(candidates, threshold_km)
summary_area = area_summary(candidates, selected_areas, threshold_km)
if recommendation_engine.startswith("GeoVision AI"):
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
# Branded institutional header matching the approved dashboard style.
brand_logo_path = BASE / "assets" / "university_logo.jpg"
brand_logo_b64 = image_to_base64(brand_logo_path)

brand_header_html = f"""<style>
.brand-banner{{box-sizing:border-box;width:100%;min-height:100px;margin:0 0 1.25rem 0;padding:12px 22px;border:1px solid rgba(75,126,198,.42);border-radius:14px;background:linear-gradient(100deg,#0a1835 0%,#07142d 52%,#061328 100%);box-shadow:inset 0 0 0 1px rgba(255,255,255,.015);display:flex;align-items:center;justify-content:space-between;gap:24px;overflow:hidden}}
.brand-left{{display:flex;align-items:center;gap:18px;min-width:0;flex:1 1 auto}}
.brand-logo{{width:74px;height:74px;min-width:74px;border-radius:50%;object-fit:cover;border:2px solid rgba(142,170,255,.72);box-shadow:0 0 0 4px rgba(255,255,255,.035)}}
.brand-copy{{min-width:0}}
.brand-university{{margin:0;color:#fff;font-size:1.55rem;line-height:1.12;font-weight:800;letter-spacing:-.015em;white-space:nowrap}}
.brand-team{{margin:.35rem 0 0 0;color:#f3f6ff;font-size:1.08rem;line-height:1.2;font-weight:500}}
.brand-network-svg{{width:38%;max-width:540px;min-width:330px;height:76px;flex:0 0 auto}}
@media(max-width:1050px){{.brand-university{{font-size:1.25rem;white-space:normal}}.brand-network-svg{{width:34%;min-width:240px}}}}
@media(max-width:760px){{.brand-banner{{padding:14px 16px}}.brand-network-svg{{display:none}}.brand-logo{{width:64px;height:64px;min-width:64px}}.brand-university{{font-size:1.15rem}}.brand-team{{font-size:.95rem}}}}
</style><div class="brand-banner"><div class="brand-left"><img class="brand-logo" src="data:image/jpeg;base64,{brand_logo_b64}" alt="University of Technology (Yatanarpon Cyber City) logo"><div class="brand-copy"><div class="brand-university">University of Technology (Yatanarpon Cyber City)</div><div class="brand-team">Team GeoVisionaries</div></div></div><svg class="brand-network-svg" viewBox="0 0 540 90" preserveAspectRatio="xMidYMid meet" aria-hidden="true"><defs><filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><g fill="none" stroke="#2e77c7" stroke-width="1" opacity=".68"><path d="M15 38 L90 66 L145 49 L220 22 L292 54 L367 29 L438 64 L520 25"/><path d="M90 66 L150 32 L292 54 L345 18 L438 64"/><path d="M145 49 L220 22 L292 54 L367 29 L438 64 L520 25"/><path d="M15 38 L150 32 L220 22"/></g><g fill="#5aa9ff" filter="url(#glow)"><circle cx="15" cy="38" r="3"/><circle cx="90" cy="66" r="3"/><circle cx="145" cy="49" r="3"/><circle cx="150" cy="32" r="3"/><circle cx="220" cy="22" r="3"/><circle cx="292" cy="54" r="3"/><circle cx="345" cy="18" r="3"/><circle cx="367" cy="29" r="3"/><circle cx="438" cy="64" r="3"/><circle cx="520" cy="25" r="3"/></g></svg></div>"""

st.markdown(brand_header_html, unsafe_allow_html=True)

st.title("📡 GeoVision AI — Yangon Telecom Coverage & Resilience Planner")
st.caption(
    "Multi-model AI: recommend new tower locations and estimate Flood, Earthquake, Cyclone and Compound disaster impact on existing towers."
)
st.info(
    "This is a planning tool, not a live operator network monitor. Disaster Impact AI can use GEE rainfall/terrain and cyclone archive context plus USGS earthquake events; population and telecom coverage remain geographic estimates requiring field and RF engineering validation."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Population 2020", f"{baseline_metrics.get('population_total', 0):,.0f}")
k2.metric("Mapped tower sites", f"{len(towers):,}")
k3.metric("Population within planning range", f"{baseline_metrics.get('baseline_served', 0):,.0f}", f"{baseline_metrics.get('baseline_coverage_pct', 0):.1f}%")
k4.metric("Population outside planning range", f"{baseline_metrics.get('baseline_uncovered', 0):,.0f}", f"> {service_radius_km:.1f} km")
k5.metric("Underserved local areas", f"{int((candidates.nearest_tower_km >= threshold_km).sum()):,}")

# ---------- tabs ----------
t_overview, t_population, t_gap, t_recommend, t_ai, t_disaster, t_rain, t_method = st.tabs(
    [
        "Overview",
        "Population & tower load",
        "Underserved areas",
        "Suggested tower locations",
        "AI Site Checker",
        "Disaster impact",
        "Rainfall & flood",
        "Technical details",
    ]
)

with t_overview:
    st.subheader("Where are the current coverage gaps?")
    fig = base_map(selected_areas, towers, zoom=8.45 if len(selected_areas) > 1 else 9.1)
    add_towers(fig, towers)
    add_candidates(fig, candidates, "Underserved local areas")
    add_recommendations(fig, recs)
    st.caption("Orange/red points show local areas farther from existing towers. Green numbered points show suggested locations for further field review.")
    overview_event = st.plotly_chart(fig, use_container_width=True, key="overview_gap_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
    overview_detail = st.container()
    with overview_detail:
        show_map_selection(overview_event, candidates, recs, "overview")

    st.markdown("#### Coverage need by area")
    st.dataframe(summary_area, use_container_width=True, hide_index=True)

with t_population:
    st.subheader("Where could existing tower sites be carrying the most population demand?")
    st.write(
        "The map estimates how much nearby population is associated with each mapped tower location. "
        "Use it to spot areas where infrastructure may be carrying more geographic demand. It is not a count of real subscribers or network traffic."
    )

    p_rows = []
    for area in selected_areas:
        area_towers = tower_sites_all[tower_sites_all.analysis_area == area]
        aid = AREA_ID[area]
        pop_total = float(pop_grid["population"][pop_grid["area_id"] == aid].sum())
        p_rows.append({
            "Area": area,
            "Population 2020": int(round(pop_total)),
            "Mapped tower sites": int(len(area_towers)),
            "Average population / site": int(round(pop_total / max(len(area_towers), 1))),
            "Median estimated nearby population": int(round(area_towers.estimated_population_nearest.median())),
            "Highest estimated nearby population": int(round(area_towers.estimated_population_nearest.max())),
            "Tower sites in historic flood areas": int((area_towers.flood_frequency > 0).sum()),
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
                    "<br>Technology: %{customdata[2]} | mapped cells: %{customdata[3]}"
                    "<br>Estimated nearby population: %{customdata[4]:,.0f}"
                    "<br>Population within 5 km: %{customdata[5]:,.0f}"
                    "<br>Historic flood frequency: %{customdata[6]:.0f}<extra></extra>"
                ),
                name="Estimated population near tower",
            )
        )
        st.plotly_chart(fig, use_container_width=True, config=MAP_PLOTLY_CONFIG)

    with c2:
        st.markdown("**Tower sites with the highest estimated nearby population**")
        top = towers.nlargest(25, "estimated_population_primary_5km")[[
            "tower_id", "analysis_area", "adm3_name", "radios", "cell_count",
            "estimated_population_primary_5km", "estimated_population_nearest", "flood_frequency",
        ]].copy()
        top["estimated_population_primary_5km"] = top.estimated_population_primary_5km.round(0).astype(int)
        top["estimated_population_nearest"] = top.estimated_population_nearest.round(0).astype(int)
        top.columns = [
            "Site ID", "Area", "Township", "Technology", "Mapped cells", "Population within 5 km",
            "Estimated nearby population", "Historic flood frequency",
        ]
        st.dataframe(top, use_container_width=True, hide_index=True, height=520)

    st.caption(
        "The 5 km population is the easier planning measure to compare. The broader nearby-population estimate assigns each person to the nearest mapped site even when farther away, so it highlights infrastructure scarcity rather than actual signal coverage."
    )

with t_gap:
    st.subheader("Which communities are farthest from existing towers?")
    underserved = candidates[candidates.nearest_tower_km >= threshold_km].sort_values(
        ["population_2020", "nearest_tower_km"], ascending=False
    )
    underserved = add_area_label(underserved)
    c1, c2 = st.columns([1.55, 1])
    with c1:
        fig = base_map(selected_areas, underserved if len(underserved) else candidates, zoom=8.6 if len(selected_areas) > 1 else 9.2)
        gap_source = underserved if len(underserved) else candidates
        add_candidates(fig, gap_source, "Underserved local areas")
        st.caption("Click an orange/red area to see the nearest existing tower and the reason it is flagged as underserved.")
        under_event = st.plotly_chart(fig, use_container_width=True, key="underserved_gap_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
        under_detail = st.container()
        with under_detail:
            show_nearest_tower_selection(under_event, gap_source, "underserved")
    with c2:
        st.metric("Population in underserved local areas", f"{underserved.population_2020.sum():,.0f}")
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
        top.columns = ["Area", "Township", "Ward / Village Tract", "Population 2020", "Nearest tower (km)", "Tower ID", "Tower latitude", "Tower longitude", "Disaster risk /100"]
        st.dataframe(top, use_container_width=True, hide_index=True, height=490)

    st.info(
        "Distance alone does not tell the full story. Use population together with tower distance to identify where infrastructure investment could benefit more people."
    )

with t_recommend:
    st.subheader("Where should new tower sites be investigated first?")
    st.write(
        "Green points are high-priority **planning candidates** based on tower distance, population need, disaster exposure and terrain. "
        "Select a point to see the reason for the recommendation and the next action."
    )
    with st.expander("How these suggestions are calculated", expanded=False):
        if recommendation_engine.startswith("GeoVision AI"):
            st.write(
                "GeoVision AI ranks candidate areas using coverage need, population need, rural priority, disaster safety and terrain suitability. "
                "A spacing rule then prevents suggested sites from clustering too closely."
            )
            st.caption("Technical GeoVision AI model file: `models/tower_site_xgb.json`.")
        else:
            st.write(
                "The comparison method uses weighted planning rules for coverage need, population need, rural priority, disaster safety and terrain suitability. "
                "A spacing rule then prevents suggested sites from clustering too closely."
            )

    fig = base_map(selected_areas, recs, zoom=8.6 if len(selected_areas) > 1 else 9.2)
    add_towers(fig, towers, max_points=1600)
    add_recommendations(fig, recs)
    st.caption("Select a green numbered location to see why it is recommended and what should be checked next.")
    rec_event = st.plotly_chart(fig, use_container_width=True, key="recommendation_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
    rec_detail = st.container()
    with rec_detail:
        if not show_recommendation_selection(rec_event, recs, "recommendation"):
            st.caption("Select a green location to see why a new tower is recommended there.")

    recs_display = add_area_label(recs)

    # Simple decision table for normal users. Raw model fields remain available below.
    simple = recs_display.copy()
    simple["Population Need"] = simple["population_score"].apply(_level)
    simple["Disaster Risk"] = simple["hazard_score"].apply(_level)
    simple["Recommendation"] = simple.apply(_recommendation_label, axis=1)
    simple["Nearest Tower (km)"] = simple["nearest_tower_km"].astype(float).round(1)
    simple["Overall Site Score"] = simple["suitability_score"].astype(float).round(1)
    simple = simple[[
        "rank", "adm3_name", "adm4_name", "Nearest Tower (km)",
        "Population Need", "Disaster Risk", "Overall Site Score", "Recommendation"
    ]].rename(columns={
        "rank": "Rank",
        "adm3_name": "Township",
        "adm4_name": "Ward / Village Tract",
    })
    st.markdown("#### Priority list")
    st.dataframe(simple, use_container_width=True, hide_index=True)

    export_cols = [
        "rank", "analysis_area", "adm3_name", "adm4_name", "lat", "lon", "population_2020",
        "nearest_tower_id", "nearest_tower_lat", "nearest_tower_lon", "nearest_tower_km",
        "gap_score", "population_score", "elevation_m", "elevation_score", "hazard_score", "safety_score", "suitability_score",
    ]
    for optional_col in ["ai_probability", "ai_decision", "recommendation_engine"]:
        if optional_col in recs_display.columns:
            export_cols.append(optional_col)
    technical_table = recs_display[export_cols].copy()
    technical_table = technical_table.rename(columns={"lat": "latitude", "lon": "longitude"})
    if "ai_probability" in technical_table:
        technical_table["ai_probability"] = (100 * technical_table["ai_probability"].astype(float)).round(2)
        technical_table = technical_table.rename(columns={"ai_probability": "model_recommendation_score_pct"})
    for c in ["latitude", "longitude", "nearest_tower_lat", "nearest_tower_lon"]:
        technical_table[c] = technical_table[c].astype(float).round(6)
    for c in ["nearest_tower_km", "gap_score", "population_score", "elevation_score", "hazard_score", "safety_score"]:
        technical_table[c] = technical_table[c].astype(float).round(4)
    if "elevation_m" in technical_table:
        technical_table["elevation_m"] = technical_table["elevation_m"].astype(float).round(1)
    technical_table["suitability_score"] = technical_table.suitability_score.astype(float).round(2)
    technical_table["population_2020"] = technical_table.population_2020.round(0).astype(int)

    with st.expander("Technical recommendation table", expanded=False):
        st.caption("Raw coordinates and model factors are kept here for analysts, engineers and judges who want to inspect the calculation.")
        st.dataframe(technical_table, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download suggested locations with coordinates (CSV)",
        technical_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="yangon_suggested_tower_locations.csv",
        mime="text/csv",
    )
    st.caption("Suggested coordinates are for field investigation only; they are not final approved construction coordinates.")

with t_ai:
    st.subheader("Check a proposed tower location")
    status = model_status()
    st.write(
        "Click anywhere inside the study area to check whether that location is a strong candidate for further tower planning. "
        "The result explains the main reasons in plain language."
    )
    with st.expander("About the model", expanded=False):
        st.info(
            f"GeoVision AI • {status['training_rows']} training rows • "
            f"{len(status['features'])} model factors • decision threshold {100*status['threshold']:.0f}%."
        )

    previous = st.session_state.get("ai_site_checker_point")
    clicked_point = None

    if HAS_CLICK_MAP:
        m = build_ai_click_map(selected_areas, previous)
        st.caption("Click anywhere inside the blue study boundary. You can also enter an exact coordinate below.")
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
            if st.button("Check location", type="primary", key="ai_manual_assess"):
                clicked_point = (float(manual_lat), float(manual_lon))
                st.session_state["ai_site_checker_point"] = clicked_point

    if clicked_point:
        render_ai_assessment(assess_site(clicked_point[0], clicked_point[1]), key_prefix="ai_checker")
    else:
        st.caption("No location selected yet. Click the map or enter a coordinate to check a proposed site.")


with t_disaster:
    st.subheader("GeoVision Disaster Impact AI — multi-hazard exposure for existing towers")
    st.write(
        "The old manual disaster-severity simulator has been replaced by Model 2, a multi-hazard AI system. "
        "Choose Flood/Heavy Rain, Earthquake, Cyclone, or Compound. Each module converts its hazard data into "
        "tower-level features and sends them to a trained XGBoost impact model; Compound AI then learns from the "
        "three hazard-model outputs together."
    )

    hazard_label_to_type = {
        "Flood / Heavy Rain": "flood",
        "Earthquake": "earthquake",
        "Cyclone": "cyclone",
        "Compound": "compound",
    }
    hazard_label = st.selectbox(
        "AI disaster model",
        list(hazard_label_to_type.keys()),
        index=0,
        help="Compound AI runs all three hazard modules and combines their tower-level scores.",
    )
    selected_hazard_type = hazard_label_to_type[hazard_label]
    hz_status = hazard_model_status(selected_hazard_type)

    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Selected AI model", hazard_label)
    z2.metric("Model type", hz_status["model_type"])
    z3.metric("Training rows", f"{hz_status['training_rows']:,}")
    z4.metric("Training towers", f"{hz_status['training_towers']:,}")

    feature_contracts = {
        "flood": "GEE GSMaP rainfall • SRTM elevation/slope • historic flood susceptibility",
        "earthquake": "USGS recent earthquake event signal • historic seismic exposure • tower isolation/redundancy",
        "cyclone": "GEE NOAA IBTrACS track/wind context • historic cyclone exposure • elevation/flood/isolation",
        "compound": "Flood AI score • Earthquake AI score • Cyclone AI score • tower isolation",
    }
    st.caption(
        f"**{hazard_label} feature contract:** {feature_contracts[selected_hazard_type]}. "
        "Scores are planning exposure/impact scores, not calibrated physical disaster probabilities."
    )

    h1, h2 = st.columns([1.2, 1])
    with h1:
        hazard_mode_label = st.radio(
            "Disaster AI data source",
            ["Auto: live sources then local fallback", "Live external data only", "Local cached demo"],
            horizontal=False,
            help=(
                "Live Flood uses Google Earth Engine GSMaP/SRTM. Live Cyclone uses GEE NOAA IBTrACS. "
                "Live Earthquake uses the USGS earthquake catalog because Earth Engine is not a real-time seismic-event source. "
                "Auto keeps the demo working if a live source is unavailable."
            ),
        )
    with h2:
        st.markdown("**Model 2 structure**")
        st.caption("Flood AI • Earthquake AI • Cyclone AI → Compound AI → tower exposure → population coverage impact")

    gee_project_id = None
    gee_service_json = None
    try:
        gee_section = st.secrets.get("gee", {})
        gee_project_id = gee_section.get("project_id")
        gee_service_json = gee_section.get("service_account_json")
    except Exception:
        pass

    mode_map = {
        "Auto: live sources then local fallback": "auto",
        "Live external data only": "gee",
        "Local cached demo": "local",
    }

    selected_hazard_towers = tower_sites_all[tower_sites_all.analysis_area.isin(selected_areas)].copy()
    run_col, info_col = st.columns([0.8, 2.2])
    with run_col:
        run_hazard = st.button("Run Disaster Impact AI", type="primary", key="run_hazard_ai")
    with info_col:
        st.caption(f"Will run {hazard_label} AI for {len(selected_hazard_towers):,} existing tower sites in the selected analysis area(s).")

    if run_hazard:
        with st.spinner(f"Running {hazard_label} AI and preparing tower exposure..."):
            try:
                hazard_result, hazard_run = run_hazard_ai(
                    selected_hazard_towers,
                    mode=mode_map[hazard_mode_label],
                    project_id=gee_project_id,
                    service_account_json=gee_service_json,
                    hazard_type=selected_hazard_type,
                )
                st.session_state["hazard_ai_result"] = hazard_result
                st.session_state["hazard_ai_run"] = hazard_run
                st.session_state["hazard_ai_areas"] = tuple(selected_areas)
                st.session_state["hazard_ai_type"] = selected_hazard_type
            except Exception as exc:
                st.session_state.pop("hazard_ai_result", None)
                st.session_state.pop("hazard_ai_run", None)
                st.session_state.pop("hazard_ai_type", None)
                st.error(f"Disaster Impact AI could not run: {type(exc).__name__}: {exc}")

    hazard_result = st.session_state.get("hazard_ai_result")
    hazard_run = st.session_state.get("hazard_ai_run", {})
    hazard_areas = st.session_state.get("hazard_ai_areas")
    hazard_saved_type = st.session_state.get("hazard_ai_type")

    if hazard_result is not None and hazard_areas == tuple(selected_areas) and hazard_saved_type == selected_hazard_type:
        run_mode = hazard_run.get("mode")
        if run_mode in {"gee", "live"}:
            st.success(hazard_run.get("message", "Live disaster data used."))
        elif run_mode == "mixed":
            st.warning("Compound AI used a mixture of live and fallback data sources. Expand the source details below to see each submodel.")
        elif not hazard_run.get("ok", True):
            st.warning(hazard_run.get("message", "Live source unavailable; local fallback used."))
        else:
            st.info(hazard_run.get("message", "Local cached demo features used."))

        if selected_hazard_type == "earthquake" and run_mode in {"live", "gee"}:
            e1, e2, e3 = st.columns(3)
            e1.metric("Recent events queried", f"{hazard_run.get('event_count', 0):,}")
            mag = hazard_run.get("strongest_magnitude")
            e2.metric("Strongest recent event", f"M {mag:.1f}" if isinstance(mag, (int, float)) else "None")
            e3.metric("USGS window", f"{hazard_run.get('window_days', 30)} days")
            if hazard_run.get("strongest_place"):
                st.caption(f"Strongest queried event: {hazard_run.get('strongest_place')} • {hazard_run.get('strongest_time', '')}")
        elif selected_hazard_type == "cyclone" and run_mode == "gee":
            c1, c2, c3 = st.columns(3)
            c1.metric("IBTrACS season", str(hazard_run.get("season", "—")))
            c2.metric("Track points", f"{hazard_run.get('track_points', 0):,}")
            c3.metric("Max track wind", f"{hazard_run.get('max_wind_knots', 0):.0f} kt")
            st.caption("IBTrACS is best-track/archive observational context, not a cyclone forecast feed.")
        elif selected_hazard_type == "compound":
            with st.expander("Compound AI source details", expanded=False):
                for name, run_info in hazard_run.get("submodels", {}).items():
                    st.write(f"**{name.title()} AI:** {run_info.get('message', '')}")

        hz = hazard_result.copy().sort_values("hazard_ai_score", ascending=False)
        very_high = int((hz.hazard_ai_score >= hz_status["very_high_threshold"]).sum())
        high_plus = int((hz.hazard_ai_score >= hz_status["high_threshold"]).sum())
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Towers analyzed", f"{len(hz):,}")
        m2.metric("High + very high", f"{high_plus:,}")
        m3.metric("Very high", f"{very_high:,}")
        m4.metric("Highest exposure", f"{100*hz.hazard_ai_score.max():.1f}%")
        m5.metric("Median exposure", f"{100*hz.hazard_ai_score.median():.1f}%")

        data_source = str(hz.hazard_data_source.iloc[0]) if len(hz) else ""
        data_time = str(hz.hazard_data_timestamp.iloc[0]) if len(hz) else ""
        st.caption(f"Data/model source: {data_source}" + (f" • reference timestamp: {data_time}" if data_time else ""))

        hazard_failure_threshold = st.slider(
            "Treat tower as hazard-affected at AI exposure score",
            min_value=0.30, max_value=0.90, value=float(hz_status["high_threshold"]), step=0.05,
            key=f"hazard_ai_failure_threshold_{selected_hazard_type}",
            help="Planning sensitivity threshold only. The AI score is not a calibrated tower-failure probability.",
        )
        risk_for_coverage = tower_sites_all.copy()
        score_lookup = hz.set_index("tower_id")["hazard_ai_score"]
        risk_for_coverage["scenario_risk"] = risk_for_coverage["tower_id"].map(score_lookup).fillna(0.0)
        risk_for_coverage["risk_class"] = "Not assessed"
        hazard_metrics, hazard_load = simulate_population_coverage(
            pop_grid,
            risk_for_coverage,
            [AREA_ID[a] for a in selected_areas],
            selected_areas,
            service_radius_km,
            hazard_failure_threshold,
        )
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("AI-affected towers", f"{hazard_metrics.get('selected_failed_towers', 0):,}")
        q2.metric("People initially affected", f"{hazard_metrics.get('population_directly_affected', 0):,.0f}")
        q3.metric("People rerouted", f"{hazard_metrics.get('population_rerouted', 0):,.0f}")
        q4.metric("Potentially losing access", f"{hazard_metrics.get('population_losing_coverage', 0):,.0f}")
        q5.metric(
            "Coverage after hazard",
            f"{hazard_metrics.get('post_coverage_pct', 0):.1f}%",
            f"{hazard_metrics.get('post_coverage_pct', 0)-hazard_metrics.get('baseline_coverage_pct', 0):+.1f} pp",
        )
        st.caption(
            "Population impact uses the existing nearest-surviving-tower planning model. It is a geographic resilience estimate, not RF propagation or observed subscriber handover."
        )

        fig = base_map(selected_areas, hz, zoom=8.6 if len(selected_areas) > 1 else 9.2)
        draw = hz.head(2500).copy()
        critical = hz[hz.hazard_ai_score >= hz_status["high_threshold"]]
        if len(critical):
            draw = pd.concat([draw, critical]).drop_duplicates("tower_id")

        # Build hazard-specific hover text while keeping the same map mechanics.
        if selected_hazard_type == "flood":
            draw["context1"] = draw.get("rain_30d_mm", pd.Series(np.nan, index=draw.index))
            draw["context2"] = draw.get("rain_72h_mm", pd.Series(np.nan, index=draw.index))
            context_template = "<br>30-day rain: %{customdata[4]:.1f} mm<br>72-hour rain: %{customdata[5]:.1f} mm"
        elif selected_hazard_type == "earthquake":
            draw["context1"] = draw.get("earthquake_history_score", pd.Series(np.nan, index=draw.index))
            draw["context2"] = draw.get("event_intensity", pd.Series(np.nan, index=draw.index))
            context_template = "<br>Historical seismic exposure: %{customdata[4]:.1%}<br>Recent-event signal: %{customdata[5]:.1%}"
        elif selected_hazard_type == "cyclone":
            draw["context1"] = draw.get("cyclone_history_score", pd.Series(np.nan, index=draw.index))
            draw["context2"] = draw.get("event_intensity", pd.Series(np.nan, index=draw.index))
            context_template = "<br>Historical cyclone exposure: %{customdata[4]:.1%}<br>Track/wind signal: %{customdata[5]:.1%}"
        else:
            draw["context1"] = draw.get("flood_ai_score", pd.Series(np.nan, index=draw.index))
            draw["context2"] = draw.get("earthquake_ai_score", pd.Series(np.nan, index=draw.index))
            context_template = "<br>Flood AI: %{customdata[4]:.1%}<br>Earthquake AI: %{customdata[5]:.1%}<br>Cyclone AI: %{customdata[6]:.1%}"

        if selected_hazard_type == "compound":
            customdata = np.column_stack([
                draw.tower_id, draw.adm3_name, draw.hazard_ai_score, draw.hazard_class,
                draw.context1, draw.context2, draw.get("cyclone_ai_score", pd.Series(np.nan, index=draw.index)),
            ])
        else:
            customdata = np.column_stack([
                draw.tower_id, draw.adm3_name, draw.hazard_ai_score, draw.hazard_class,
                draw.context1, draw.context2,
            ])
        fig.add_trace(
            go.Scattermap(
                lat=draw.lat,
                lon=draw.lon,
                mode="markers",
                marker={
                    "size": np.where(draw.hazard_ai_score >= hz_status["very_high_threshold"], 11, 7),
                    "color": draw.hazard_ai_score,
                    "cmin": 0, "cmax": 1, "colorscale": "Turbo", "showscale": True,
                    "colorbar": {"title": {"text": f"{hazard_label} AI", "side": "right"}, "x": 1.02, "len": 0.68},
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>Tower %{customdata[0]}</b><br>%{customdata[1]}"
                    f"<br>{hazard_label} AI exposure: %{{customdata[2]:.1%}} (%{{customdata[3]}})"
                    + context_template + "<extra></extra>"
                ),
                name=f"{hazard_label} AI exposure",
            )
        )
        st.plotly_chart(fig, use_container_width=True, config=MAP_PLOTLY_CONFIG)

        base_cols = ["tower_id", "analysis_area", "adm3_name", "networks", "radios", "hazard_ai_pct", "hazard_class"]
        if selected_hazard_type == "flood":
            extra_cols = ["rain_30d_mm", "rain_72h_mm", "elevation_m", "slope_deg", "flood_history_score"]
        elif selected_hazard_type == "earthquake":
            extra_cols = ["earthquake_history_score", "event_intensity", "isolation_score", "radio_vulnerability"]
        elif selected_hazard_type == "cyclone":
            extra_cols = ["cyclone_history_score", "event_intensity", "elevation_risk", "flood_history_score", "isolation_score"]
        else:
            extra_cols = ["flood_ai_score", "earthquake_ai_score", "cyclone_ai_score", "isolation_score"]
        table = hz[[c for c in base_cols + extra_cols if c in hz.columns]].head(40).copy()
        table["hazard_ai_pct"] = table["hazard_ai_pct"].round(1)
        for c in table.columns:
            if c.endswith("_score") or c in {"event_intensity", "radio_vulnerability", "elevation_risk"}:
                table[c] = (100 * pd.to_numeric(table[c], errors="coerce")).round(1)
            elif c in {"rain_30d_mm", "rain_72h_mm", "elevation_m", "slope_deg"}:
                table[c] = pd.to_numeric(table[c], errors="coerce").round(1)
        rename_map = {
            "tower_id": "Tower ID", "analysis_area": "Analysis Area", "adm3_name": "Township",
            "networks": "Operator", "radios": "Technology", "hazard_ai_pct": f"{hazard_label} AI Exposure (%)",
            "hazard_class": "Exposure Class", "rain_30d_mm": "30-day Rain (mm)", "rain_72h_mm": "72-hour Rain (mm)",
            "elevation_m": "Elevation (m)", "slope_deg": "Slope (deg)", "flood_history_score": "Historic Flood (%)",
            "earthquake_history_score": "Historic Seismic Exposure (%)", "cyclone_history_score": "Historic Cyclone Exposure (%)",
            "event_intensity": "Current/Recent Event Signal (%)", "isolation_score": "Isolation Risk (%)",
            "radio_vulnerability": "Radio Vulnerability Proxy (%)", "elevation_risk": "Low-Terrain Risk (%)",
            "flood_ai_score": "Flood AI (%)", "earthquake_ai_score": "Earthquake AI (%)", "cyclone_ai_score": "Cyclone AI (%)",
        }
        table = table.rename(columns=rename_map)
        st.markdown(f"**Highest-exposure current tower sites — {hazard_label} AI**")
        st.dataframe(table, use_container_width=True, hide_index=True, height=470)

        with st.expander("Disaster Impact AI limitations and interpretation", expanded=False):
            for item in hz_status["limitations"]:
                st.write(f"- {item}")
    else:
        st.info(
            f"Run {hazard_label} AI to generate tower-level disaster exposure. Auto mode tries the appropriate live source first and falls back to the project's cached data when necessary."
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
    latest = latest.rename(columns={
        "adm2_name": "Yangon Subregion",
        "date": "Date",
        "rfh": "10-day Rainfall (mm)",
        "rfh_avg": "10-day Long-term Avg (mm)",
        "rfq": "10-day Anomaly (%)",
        "r1h": "1-month Rainfall (mm)",
        "r1h_avg": "1-month Long-term Avg (mm)",
        "r1q": "1-month Anomaly (%)",
        "r1h_percentile": "1-month Rainfall Percentile",
        "r3h": "3-month Rainfall (mm)",
        "r3q": "3-month Anomaly (%)",
    })
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
        "**Multi-model GeoVision AI:** Model 1 (`models/tower_site_xgb.json`) recommends new tower locations. "
        "Model 2 is now a multi-hazard Disaster Impact AI with four XGBoost modules: Flood/Heavy Rain (`hazard_flood_xgb.json`), "
        "Earthquake (`hazard_earthquake_xgb.json`), Cyclone (`hazard_cyclone_xgb.json`) and Compound (`hazard_compound_xgb.json`). "
        "Flood uses live GEE GSMaP/SRTM when available; Cyclone uses GEE NOAA IBTrACS observational track/wind context; Earthquake uses recent USGS events because GEE is not the appropriate real-time earthquake-event source. "
        "All four are hackathon/MVP exposure models with pseudo-label limitations and should not be presented as calibrated physical disaster probabilities."
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

        **3. GeoVision Disaster Impact AI (Model 2 — multi-hazard)**
        - **Flood / Heavy Rain AI:** live GEE mode queries JAXA GSMaP rainfall plus SRTM elevation/slope and combines them with historic flood susceptibility.
        - **Earthquake AI:** live mode queries recent USGS earthquake events, converts magnitude/depth/distance into a tower-level event signal, and combines it with the project's historic seismic exposure and tower vulnerability proxies.
        - **Cyclone AI:** live GEE mode queries NOAA IBTrACS best-track/archive observations and derives a tower-level track/wind signal, combined with historic cyclone exposure, terrain/flood context and tower isolation. IBTrACS is observational archive context, not a forecast feed.
        - **Compound AI:** a trained meta-model combines the Flood AI, Earthquake AI and Cyclone AI scores plus tower isolation to estimate multi-hazard planning exposure.
        - Earthquake, cyclone and compound targets are transparent pseudo-labels. Replace them with verified event/outage labels before making operational probability claims.

        **4. AI-driven disaster coverage impact**
        - The selected AI hazard score replaces the old manual severity scenario risk.
        - Towers above the selected AI exposure threshold are treated as unavailable for planning sensitivity analysis.
        - Population is re-routed to the nearest surviving alternative among its 10 nearest precomputed sites.
        - If no surviving alternative is inside the planning service radius, that population is counted as potentially losing coverage.
        """
    )

    st.markdown("#### Critical interpretation")
    st.error(
        "Estimated population per site is NOT the number of real customers connected to that tower. Actual users require operator subscriber/traffic data. "
        "Likewise, the planning radius is NOT RF propagation. A production model should add antenna frequency, height, azimuth, transmit power, terrain/DEM, buildings, handover/traffic logs, backup power and tower outage history."
    )

# ---------- footer ----------
st.markdown("---")
st.caption("Developed by Team GeoVisionaries, University of Technology (Yatanarpon Cyber City).")

