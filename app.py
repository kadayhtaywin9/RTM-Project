from __future__ import annotations

import base64
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from engine.hazard_engine import get_hazard_engine
from hazard_ai import hazard_model_status
from site_ai import assess_site, model_status, recommend_sites_xgb
from ui.hazard_results import render_hazard_results

try:
    import folium
    from streamlit_folium import st_folium
    HAS_CLICK_MAP = True
except ImportError:
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
DYNAMIC_WORLD_CLASSES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Shrub and scrub",
    6: "Built area",
    7: "Bare ground",
    8: "Snow and ice",
}

st.set_page_config(
    page_title="GeoVision AI | Yangon Network Resilience",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAP_STYLE = "open-street-map"

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


def inject_global_styles():
    """Apply the compact, restrained GeoVision planning-dashboard styles."""
    st.markdown(
        """
        <style>
        :root {
            --gv-bg: #07111f;
            --gv-bg-soft: #0a1628;
            --gv-surface: #0f2038;
            --gv-surface-strong: #132945;
            --gv-border: rgba(148, 180, 224, 0.18);
            --gv-text: #f4f8ff;
            --gv-muted: #9fb0c8;
            --gv-blue: #5b82ff;
            --gv-cyan: #2dd4bf;
            --gv-amber: #f59e0b;
            --gv-red: #f43f5e;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 83% -8%, rgba(49, 94, 211, 0.20), transparent 31rem),
                radial-gradient(circle at 38% 10%, rgba(45, 212, 191, 0.06), transparent 25rem),
                var(--gv-bg);
            color: var(--gv-text);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1480px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(19, 41, 69, 0.98), rgba(8, 20, 37, 0.99));
            border-right: 1px solid var(--gv-border);
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            color: #c8d5e7;
        }

        .gv-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0 0 0.55rem;
            padding: 0.75rem;
            border: 1px solid rgba(91, 130, 255, 0.25);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(91, 130, 255, 0.14), rgba(45, 212, 191, 0.06));
        }

        .gv-sidebar-mark {
            display: grid;
            width: 2.25rem;
            height: 2.25rem;
            flex: 0 0 2.25rem;
            place-items: center;
            border-radius: 11px;
            background: linear-gradient(135deg, var(--gv-blue), var(--gv-cyan));
            color: #06111e;
            font-weight: 900;
            box-shadow: 0 8px 22px rgba(45, 212, 191, 0.18);
        }

        .gv-sidebar-title {
            color: var(--gv-text);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .gv-sidebar-subtitle {
            margin-top: 0.2rem;
            color: var(--gv-muted);
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .gv-scenario-card {
            margin-top: 0.55rem;
            padding: 0.75rem 0.8rem;
            border: 1px solid var(--gv-border);
            border-radius: 13px;
            background: rgba(7, 17, 31, 0.58);
        }

        .gv-scenario-label {
            margin-bottom: 0.45rem;
            color: #7f96b5;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .gv-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .gv-chip {
            display: inline-flex;
            align-items: center;
            min-height: 1.65rem;
            padding: 0.24rem 0.52rem;
            border: 1px solid rgba(148, 180, 224, 0.16);
            border-radius: 999px;
            background: rgba(91, 130, 255, 0.09);
            color: #cfe0f7;
            font-size: 0.69rem;
            font-weight: 650;
        }

        .gv-chip--active {
            border-color: rgba(45, 212, 191, 0.30);
            background: rgba(45, 212, 191, 0.10);
            color: #8df3e2;
        }

        .gv-hero {
            position: relative;
            isolation: isolate;
            overflow: hidden;
            margin: 0 0 0.85rem;
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(91, 130, 255, 0.34);
            border-radius: 20px;
            background:
                radial-gradient(circle at 83% 22%, rgba(45, 212, 191, 0.17), transparent 18rem),
                linear-gradient(120deg, rgba(19, 41, 69, 0.98), rgba(8, 24, 45, 0.98));
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .gv-hero::after {
            content: "";
            position: absolute;
            z-index: -1;
            right: -4rem;
            bottom: -8rem;
            width: 25rem;
            height: 25rem;
            border: 1px solid rgba(91, 130, 255, 0.18);
            border-radius: 50%;
            box-shadow: 0 0 0 3.3rem rgba(91, 130, 255, 0.035), 0 0 0 7rem rgba(45, 212, 191, 0.025);
        }

        .gv-hero-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }

        .gv-institution {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            min-width: 0;
        }

        .gv-logo {
            width: 42px;
            height: 42px;
            flex: 0 0 42px;
            border: 1px solid rgba(180, 205, 244, 0.52);
            border-radius: 50%;
            object-fit: cover;
            box-shadow: 0 0 0 4px rgba(91, 130, 255, 0.08);
        }

        .gv-institution-name {
            overflow: hidden;
            color: #dce8f9;
            font-size: 0.77rem;
            font-weight: 700;
            line-height: 1.2;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .gv-team {
            color: #7f96b5;
            font-size: 0.69rem;
        }

        .gv-mode {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            flex: 0 0 auto;
            padding: 0.38rem 0.68rem;
            border: 1px solid rgba(45, 212, 191, 0.24);
            border-radius: 999px;
            background: rgba(45, 212, 191, 0.08);
            color: #91f2e3;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .gv-mode-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: var(--gv-cyan);
            box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.10), 0 0 12px rgba(45, 212, 191, 0.75);
        }

        .gv-eyebrow {
            margin-bottom: 0.35rem;
            color: #7fdccb;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .gv-hero h1 {
            max-width: 900px;
            margin: 0;
            color: #ffffff;
            font-size: clamp(1.8rem, 3.2vw, 2.85rem);
            font-weight: 850;
            letter-spacing: -0.035em;
            line-height: 1.04;
        }

        .gv-hero h1 span {
            color: #7da2ff;
        }

        .gv-hero-copy {
            max-width: 850px;
            margin: 0.65rem 0 0;
            color: #b5c5db;
            font-size: 0.93rem;
            line-height: 1.52;
        }

        .gv-hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.85rem;
        }

        .gv-hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.34rem 0.6rem;
            border: 1px solid rgba(148, 180, 224, 0.16);
            border-radius: 999px;
            background: rgba(5, 16, 30, 0.42);
            color: #d4e1f1;
            font-size: 0.71rem;
            font-weight: 650;
        }

        .gv-hero-pill b { color: #ffffff; }

        .gv-scope {
            margin: 0 0 0.8rem;
            border: 1px solid var(--gv-border);
            border-radius: 12px;
            background: rgba(15, 32, 56, 0.58);
            color: #aebed3;
            font-size: 0.78rem;
        }

        .gv-scope summary {
            padding: 0.55rem 0.75rem;
            color: #cbd8e9;
            cursor: pointer;
            font-weight: 700;
        }

        .gv-scope p {
            margin: 0;
            padding: 0 0.75rem 0.7rem;
            line-height: 1.48;
        }

        .gv-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.65rem;
            margin: 0 0 0.9rem;
        }

        .gv-kpi {
            position: relative;
            overflow: hidden;
            min-height: 104px;
            padding: 0.8rem 0.85rem 0.72rem;
            border: 1px solid var(--gv-border);
            border-radius: 15px;
            background: linear-gradient(155deg, rgba(19, 41, 69, 0.92), rgba(11, 28, 49, 0.92));
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
        }

        .gv-kpi::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: var(--gv-accent, var(--gv-blue));
        }

        .gv-kpi-label {
            min-height: 2.2em;
            color: #94a8c3;
            font-size: 0.70rem;
            font-weight: 750;
            letter-spacing: 0.025em;
            line-height: 1.15;
            text-transform: uppercase;
        }

        .gv-kpi-value {
            margin-top: 0.25rem;
            color: #ffffff;
            font-size: clamp(1.25rem, 2vw, 1.75rem);
            font-variant-numeric: tabular-nums;
            font-weight: 790;
            letter-spacing: -0.025em;
            line-height: 1.05;
        }

        .gv-kpi-detail {
            margin-top: 0.34rem;
            color: #8297b4;
            font-size: 0.69rem;
            line-height: 1.2;
        }

        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-bottom: 0.75rem;
            padding: 0.32rem;
            border: 1px solid var(--gv-border);
            border-radius: 14px;
            background: rgba(10, 24, 43, 0.86);
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.45rem;
            flex: 0 1 auto;
            padding: 0 0.72rem;
            border-radius: 10px;
            color: #9fb0c8;
            font-size: 0.78rem;
            font-weight: 720;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(91, 130, 255, 0.26), rgba(45, 212, 191, 0.13));
            color: #ffffff;
            box-shadow: inset 0 0 0 1px rgba(125, 162, 255, 0.25);
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        div[data-testid="stMetric"] {
            min-height: 104px;
            padding: 0.75rem 0.8rem;
            border: 1px solid var(--gv-border);
            border-radius: 14px;
            background: linear-gradient(155deg, rgba(19, 41, 69, 0.88), rgba(11, 28, 49, 0.88));
        }

        div[data-testid="stMetric"] label {
            color: #9fb0c8;
            font-size: 0.73rem;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
            font-variant-numeric: tabular-nums;
        }

        div[data-testid="stAlert"],
        details[data-testid="stExpander"] {
            border-radius: 13px;
            border-color: var(--gv-border);
        }

        div[data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--gv-border);
            border-radius: 16px;
            background: rgba(10, 24, 43, 0.72);
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.16);
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.6rem;
            border-radius: 11px;
            border-color: rgba(125, 162, 255, 0.35);
            font-weight: 760;
        }

        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, #5078ef, #297f91);
            box-shadow: 0 10px 24px rgba(58, 105, 210, 0.22);
        }

        h2, h3, h4 {
            color: #eef5ff;
            letter-spacing: -0.018em;
        }

        hr {
            border-color: var(--gv-border);
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] { padding-top: 0.75rem; }
            .gv-hero { padding: 1rem; border-radius: 16px; }
            .gv-institution-name { white-space: normal; }
            .gv-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 560px) {
            .gv-mode { display: none; }
            .gv-hero h1 { font-size: 1.72rem; }
            .gv-kpi-grid { grid-template-columns: 1fr; }
            .stTabs [data-baseweb="tab"] { padding: 0 0.55rem; font-size: 0.74rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # A restrained presentation layer keeps the dashboard feeling like a
    # professional planning tool rather than a promotional template.
    st.markdown(
        """
        <style>
        :root {
            --gv-bg: #f5f7fa;
            --gv-bg-soft: #eef2f6;
            --gv-surface: #ffffff;
            --gv-surface-strong: #f8fafc;
            --gv-border: #dce3eb;
            --gv-text: #172033;
            --gv-muted: #64748b;
            --gv-blue: #2563eb;
            --gv-cyan: #0f766e;
            --gv-amber: #d97706;
            --gv-red: #dc2626;
        }

        .stApp {
            background: var(--gv-bg);
            color: var(--gv-text);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--gv-border);
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            color: #475569;
        }

        .gv-sidebar-brand {
            gap: 0.65rem;
            margin-bottom: 0.65rem;
            padding: 0.65rem 0;
            border: 0;
            border-bottom: 1px solid var(--gv-border);
            border-radius: 0;
            background: transparent;
        }

        .gv-sidebar-mark {
            width: 2rem;
            height: 2rem;
            flex-basis: 2rem;
            border-radius: 8px;
            background: #163a63;
            color: #ffffff;
            box-shadow: none;
        }

        .gv-sidebar-title,
        .gv-institution-name {
            color: var(--gv-text);
        }

        .gv-sidebar-subtitle,
        .gv-team {
            color: var(--gv-muted);
        }

        .gv-scenario-card {
            border-color: var(--gv-border);
            border-radius: 9px;
            background: #f8fafc;
        }

        .gv-scenario-label {
            color: #64748b;
        }

        .gv-scenario-summary {
            color: #334155;
            font-size: 0.78rem;
            line-height: 1.5;
        }

        .gv-scenario-summary strong {
            color: #172033;
            font-weight: 700;
        }

        .gv-chip,
        .gv-chip--active {
            border-color: #d8e0e9;
            background: #ffffff;
            color: #475569;
            font-weight: 600;
        }

        .gv-chip--active {
            border-color: #b9cdf5;
            background: #eff6ff;
            color: #1d4ed8;
        }

        .gv-hero {
            margin-bottom: 0.75rem;
            padding: 1rem 1.15rem;
            border: 1px solid var(--gv-border);
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        }

        .gv-hero::after,
        .gv-mode-dot {
            display: none;
        }

        .gv-hero-top {
            margin-bottom: 0.7rem;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid #edf1f5;
        }

        .gv-logo {
            width: 38px;
            height: 38px;
            flex-basis: 38px;
            border: 1px solid #cbd5e1;
            box-shadow: none;
        }

        .gv-mode {
            padding: 0;
            border: 0;
            border-radius: 0;
            border-color: #cbd5e1;
            background: transparent;
            color: #475569;
            font-weight: 600;
            letter-spacing: 0;
            text-transform: none;
        }

        .gv-eyebrow {
            margin-bottom: 0.2rem;
            color: #2563eb;
            font-size: 0.7rem;
            font-weight: 650;
            letter-spacing: 0.04em;
            text-transform: none;
        }

        .gv-hero h1 {
            max-width: none;
            color: var(--gv-text);
            font-size: clamp(1.65rem, 2.4vw, 2.2rem);
            font-weight: 750;
            letter-spacing: -0.025em;
            line-height: 1.1;
        }

        .gv-hero h1 span {
            color: inherit;
        }

        .gv-hero-copy {
            max-width: 920px;
            margin-top: 0.45rem;
            color: #526176;
            font-size: 0.9rem;
        }

        .gv-hero-pills {
            margin-top: 0.7rem;
        }

        .gv-hero-pill {
            padding: 0.28rem 0.5rem;
            border-color: #dce3eb;
            background: #f8fafc;
            color: #526176;
            font-weight: 600;
        }

        .gv-hero-pill b {
            color: #26364d;
        }

        .gv-hero-meta {
            margin-top: 0.65rem;
            color: #64748b;
            font-size: 0.76rem;
            font-weight: 600;
        }

        .gv-hero-meta span + span::before {
            content: "·";
            margin: 0 0.55rem;
            color: #a3afbf;
        }

        .gv-scope {
            border-color: var(--gv-border);
            border-radius: 9px;
            background: #ffffff;
            color: #64748b;
        }

        .gv-scope summary {
            color: #475569;
        }

        .gv-kpi-grid {
            gap: 0.6rem;
        }

        .gv-kpi {
            min-height: 94px;
            padding: 0.72rem 0.8rem;
            border-color: var(--gv-border);
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .gv-kpi::before {
            display: none;
        }

        .gv-kpi-label {
            color: #64748b;
        }

        .gv-kpi-value {
            color: #172033;
            font-weight: 730;
        }

        .gv-kpi-detail {
            color: #7b8798;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.15rem;
            padding: 0.25rem;
            border-color: var(--gv-border);
            border-radius: 10px;
            background: #ffffff;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 7px;
            color: #64748b;
            font-weight: 650;
        }

        .stTabs [aria-selected="true"] {
            background: transparent;
            color: #1d4ed8;
            box-shadow: inset 0 -2px 0 #2563eb;
        }

        div[data-testid="stMetric"] {
            border-color: var(--gv-border);
            border-radius: 10px;
            background: #ffffff;
        }

        div[data-testid="stMetric"] label {
            color: #64748b;
        }

        div[data-testid="stMetricValue"] {
            color: #172033;
        }

        div[data-testid="stAlert"],
        details[data-testid="stExpander"] {
            border-radius: 9px;
        }

        div[data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            border-color: var(--gv-border);
            border-radius: 10px;
            background: #ffffff;
            box-shadow: none;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border-color: #cbd5e1;
            font-weight: 650;
            box-shadow: none;
        }

        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            border-color: #1d4ed8;
            background: #2563eb;
            box-shadow: none;
        }

        h2, h3, h4 {
            color: #172033;
            letter-spacing: -0.012em;
        }

        hr {
            border-color: var(--gv-border);
        }

        .gv-evidence {
            padding: 0.85rem 1rem;
            border: 1px solid #dbe3ed;
            border-left: 3px solid #64748b;
            border-radius: 8px;
            background: #ffffff;
            margin-top: 0.45rem;
        }
        .gv-evidence-live { border-left-color: #167d65; }
        .gv-evidence-mixed, .gv-evidence-fallback { border-left-color: #b7791f; }
        .gv-evidence-label { font-size: 0.88rem; font-weight: 700; color: #172033; }
        .gv-research-label {
            margin-left: 0.75rem; padding: 0.16rem 0.42rem; border-radius: 4px;
            background: #f1f5f9; color: #475569; font-size: 0.69rem;
        }
        .gv-evidence p { font-size: 0.82rem; margin: 0.4rem 0 0.25rem; color: #475569; }
        .gv-evidence small { color: #64748b; font-size: 0.72rem; }
        .gv-empty-state {
            border: 1px dashed #cbd5e1; border-radius: 10px; padding: 1.6rem;
            margin-top: 1rem; background: #ffffff;
        }
        .gv-empty-state h4 { margin: 0 0 0.6rem; }
        .gv-empty-state p { max-width: 680px; color: #475569; font-size: 0.9rem; }
        .gv-empty-state small { color: #64748b; }
        @media (max-width: 720px) {
            .gv-evidence { padding: 0.75rem; }
            .gv-research-label { display: inline-block; margin: 0.2rem 0 0.2rem 0.5rem; }
        }

        .gv-campus-header {
            position: relative;
            isolation: isolate;
            display: flex;
            align-items: center;
            box-sizing: border-box;
            min-height: 70px;
            margin-bottom: 0.75rem;
            padding: 0.6rem 1rem;
            overflow: hidden;
            border: 1px solid #1d314f;
            border-radius: 10px;
            background: #091426;
            color: #f8fafc;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }

        .gv-campus-brand {
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            max-width: 58%;
            min-width: 0;
        }

        .gv-campus-logo,
        .gv-campus-logo-fallback {
            width: 48px;
            height: 48px;
            flex: 0 0 48px;
            border: 1px solid rgba(191, 219, 254, 0.8);
            border-radius: 50%;
            object-fit: cover;
            background: #ffffff;
        }

        .gv-campus-logo-fallback {
            display: grid;
            place-items: center;
            color: #173b68;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .gv-campus-copy {
            min-width: 0;
        }

        .gv-campus-name {
            color: #f8fafc;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.3;
        }

        .gv-campus-team {
            margin-top: 0.18rem;
            color: #c6d2e1;
            font-size: 0.72rem;
            line-height: 1.3;
        }

        .gv-campus-network {
            position: absolute;
            z-index: 1;
            inset: 0 0 0 52%;
            pointer-events: none;
        }

        .gv-campus-network svg {
            display: block;
            width: 100%;
            height: 100%;
        }

        @media (max-width: 720px) {
            .gv-campus-brand { max-width: 72%; }
            .gv-campus-network { left: 58%; opacity: 0.5; }
        }

        @media (max-width: 520px) {
            .gv-campus-header { min-height: 80px; padding: 0.65rem 0.8rem; }
            .gv-campus-brand { max-width: 85%; }
            .gv-campus-logo,
            .gv-campus-logo-fallback { width: 42px; height: 42px; flex-basis: 42px; }
            .gv-campus-name { font-size: 0.76rem; }
            .gv-campus-network { left: 62%; opacity: 0.32; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(selected_areas, tower_count: int):
    area_text = "All 4 project areas" if len(selected_areas) == len(ANALYSIS_AREAS) else ", ".join(selected_areas)
    st.markdown(
        f"""
        <section class="gv-hero">
            <div class="gv-eyebrow">Telecom planning and resilience</div>
            <h1>GeoVision AI</h1>
            <p class="gv-hero-copy">
                Coverage gaps, tower priorities and multi-hazard impact analysis for Yangon.
            </p>
            <div class="gv-hero-meta">
                <span>{area_text}</span>
                <span>{tower_count:,} mapped tower sites</span>
                <span>4 hazard models</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_university_header(logo_b64: str) -> None:
    logo_html = (
        f'<img class="gv-campus-logo" src="data:image/jpeg;base64,{logo_b64}" alt="">'
        if logo_b64
        else '<div class="gv-campus-logo-fallback" aria-hidden="true">UTYCC</div>'
    )
    st.markdown(
        f"""
        <header id="university-team-header" class="gv-campus-header" aria-label="University and project team">
            <div class="gv-campus-brand">
                {logo_html}
                <div class="gv-campus-copy">
                    <div class="gv-campus-name">University of Technology (Yatanarpon Cyber City)</div>
                    <div class="gv-campus-team">Team GeoVisionaries</div>
                </div>
            </div>
            <div class="gv-campus-network" aria-hidden="true">
                <svg viewBox="0 0 760 120" preserveAspectRatio="xMaxYMid slice" focusable="false">
                    <defs>
                        <linearGradient id="gv-network-fade" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0" stop-color="#38bdf8" stop-opacity="0"/>
                            <stop offset="0.36" stop-color="#38bdf8" stop-opacity="0.35"/>
                            <stop offset="1" stop-color="#2563eb" stop-opacity="0.72"/>
                        </linearGradient>
                        <radialGradient id="gv-node-glow">
                            <stop offset="0" stop-color="#dbeafe" stop-opacity="1"/>
                            <stop offset="0.25" stop-color="#60a5fa" stop-opacity="0.95"/>
                            <stop offset="1" stop-color="#2563eb" stop-opacity="0"/>
                        </radialGradient>
                    </defs>
                    <g fill="none" stroke="url(#gv-network-fade)" stroke-width="0.9" vector-effect="non-scaling-stroke">
                        <path d="M30 42 L135 58 L225 25 L315 54 L416 22 L505 47 L610 18 L735 42"/>
                        <path d="M72 86 L135 58 L244 88 L315 54 L394 94 L505 47 L565 93 L666 66 L735 42"/>
                        <path d="M225 25 L244 88 M416 22 L394 94 M505 47 L565 93 M610 18 L666 66"/>
                        <path d="M135 58 L225 25 M244 88 L394 94 M315 54 L505 47 M565 93 L735 42" opacity="0.55"/>
                    </g>
                    <g fill="#60a5fa">
                        <circle cx="30" cy="42" r="2"/><circle cx="72" cy="86" r="1.7"/>
                        <circle cx="135" cy="58" r="2.3"/><circle cx="225" cy="25" r="1.8"/>
                        <circle cx="244" cy="88" r="2"/><circle cx="315" cy="54" r="2.4"/>
                        <circle cx="394" cy="94" r="1.8"/><circle cx="416" cy="22" r="2"/>
                        <circle cx="505" cy="47" r="2.5"/><circle cx="565" cy="93" r="2"/>
                        <circle cx="610" cy="18" r="1.8"/><circle cx="666" cy="66" r="2.3"/>
                        <circle cx="735" cy="42" r="2"/>
                    </g>
                    <g fill="url(#gv-node-glow)" opacity="0.75">
                        <circle cx="315" cy="54" r="12"/><circle cx="505" cy="47" r="13"/>
                        <circle cx="666" cy="66" r="11"/>
                    </g>
                </svg>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(baseline_metrics, tower_count: int, underserved_count: int, service_radius_km: float):
    coverage_pct = float(baseline_metrics.get("baseline_coverage_pct", 0))
    cards = [
        ("Population in scope", f"{baseline_metrics.get('population_total', 0):,.0f}", "2020 planning baseline", "#5b82ff"),
        ("Mapped tower sites", f"{tower_count:,}", "Observed site proxies", "#7da2ff"),
        ("Population within range", f"{coverage_pct:.1f}%", f"{baseline_metrics.get('baseline_served', 0):,.0f} people", "#2dd4bf"),
        ("Population outside range", f"{baseline_metrics.get('baseline_uncovered', 0):,.0f}", f"Beyond {service_radius_km:.1f} km", "#f59e0b"),
        ("Underserved local areas", f"{underserved_count:,}", "Priority screening areas", "#f43f5e"),
    ]
    card_html = "".join(
        (
            f'<div class="gv-kpi" style="--gv-accent:{accent}">'
            f'<div class="gv-kpi-label">{label}</div>'
            f'<div class="gv-kpi-value">{value}</div>'
            f'<div class="gv-kpi-detail">{detail}</div>'
            "</div>"
        )
        for label, value, detail, accent in cards
    )
    st.markdown(f'<div class="gv-kpi-grid">{card_html}</div>', unsafe_allow_html=True)


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
    except (TypeError, ValueError):
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
    except (TypeError, ValueError):
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
        map={"style": MAP_STYLE, "center": center, "zoom": zoom},
        # Keep the basemap full-height and use a compact light legend inside it.
        margin={"l": 8, "r": 84, "t": 10, "b": 8},
        height=560,
        legend={
            "orientation": "h",
            "y": 0.015,
            "yanchor": "bottom",
            "x": 0.5,
            "xanchor": "center",
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "rgba(100,116,139,0.25)",
            "borderwidth": 1,
            "font": {"color": "#334155", "size": 11},
        },
    )
    return fig


def add_cyclone_forecast_track(fig, track_records):
    """Overlay the current JTWC position and forecast path when metadata is available."""
    track = pd.DataFrame(track_records or [])
    required = {"storm_id", "storm_name", "forecast_hour", "valid_time", "lat", "lon", "wind_speed"}
    if track.empty or not required.issubset(track.columns):
        return fig
    for (storm_id, storm_name), points in track.groupby(["storm_id", "storm_name"], dropna=False):
        points = points.sort_values("forecast_hour")
        customdata = np.column_stack([
            points["forecast_hour"],
            points["wind_speed"],
            points["valid_time"],
        ])
        fig.add_trace(
            go.Scattermap(
                lat=points["lat"],
                lon=points["lon"],
                mode="lines+markers",
                line={"width": 3, "color": "#0F766E"},
                marker={
                    "size": np.where(pd.to_numeric(points["forecast_hour"], errors="coerce").fillna(0).eq(0), 14, 9),
                    "color": "#0F766E",
                    "opacity": 0.92,
                },
                customdata=customdata,
                hovertemplate=(
                    f"<b>{storm_name or storm_id} · JTWC</b>"
                    "<br>Forecast lead: T+%{customdata[0]:.0f} h"
                    "<br>Wind: %{customdata[1]:.0f} kt"
                    "<br>Valid: %{customdata[2]}<extra></extra>"
                ),
                name=f"JTWC {storm_name or storm_id}",
            )
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
            marker={"size": 5, "color": "#2563EB", "opacity": 0.62},
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
                "colorscale": [[0, "#FDE68A"], [0.48, "#F59E0B"], [1, "#F43F5E"]],
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
            marker={"size": 16, "color": "#0F766E", "opacity": 0.98},
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
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (KeyError, TypeError):
            return default
    try:
        return getattr(obj, key)
    except AttributeError:
        try:
            return obj[key]
        except (IndexError, KeyError, TypeError):
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
        except TypeError:
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
                "Normalized value (0-1)": round(float(value), 4),
                "Data source / calculation": a.get("feature_sources", {}).get(feature, ""),
            })
        st.markdown("**Inputs used by the model**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        expl = pd.DataFrame(a.get("explanations", []))
        if not expl.empty:
            expl = expl[["label", "direction", "relative_impact_pct", "value"]].copy()
            expl.columns = ["Factor", "Effect on model score", "Relative model impact %", "Input value"]
            expl["Relative model impact %"] = expl["Relative model impact %"].round(1)
            expl["Input value"] = expl["Input value"].round(4)
            st.markdown("**Model explanation**")
            st.dataframe(expl, width="stretch", hide_index=True)
            st.caption(
                "Relative model impact describes this model prediction only; it is not a causal percentage."
            )

        report = _ai_report_text(a)
        st.download_button(
            "Download technical site assessment report",
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
    except (IndexError, TypeError, ValueError):
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
        map={"style": MAP_STYLE, "center": {"lat": mid_lat, "lon": mid_lon}, "zoom": zoom},
        height=420,
        margin={"l": 8, "r": 8, "t": 10, "b": 8},
        legend={
            "orientation": "h", "y": 0.015, "yanchor": "bottom",
            "x": 0.5, "xanchor": "center",
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "rgba(100,116,139,0.25)", "borderwidth": 1,
            "font": {"color": "#334155", "size": 11},
        },
    )
    st.plotly_chart(detail, width="stretch", key=f"{key_prefix}_nearest_detail", config=MAP_PLOTLY_CONFIG)

    if len(cells):
        st.markdown("**Technical cell records at the nearest mapped tower**")
        cell_cols = [x for x in ["radio", "Network", "MCC", "MNC", "TAC", "CID", "RANGE", "LAT", "LON"] if x in cells.columns]
        shown = cells[cell_cols].copy().rename(columns={"LAT": "cell_latitude", "LON": "cell_longitude", "RANGE": "reported_range_m"})
        st.dataframe(shown, width="stretch", hide_index=True, height=min(360, 70 + 35 * len(shown)))

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
    except (IndexError, TypeError, ValueError):
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
        st.dataframe(detail, width="stretch", hide_index=True)

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
            map={"style": MAP_STYLE, "center": {"lat": mid_lat, "lon": mid_lon}, "zoom": zoom},
            height=400,
            margin={"l": 8, "r": 8, "t": 10, "b": 8},
            legend={
                "orientation": "h", "y": 0.015, "yanchor": "bottom",
                "x": 0.5, "xanchor": "center",
                "bgcolor": "rgba(255,255,255,0.90)",
                "bordercolor": "rgba(100,116,139,0.25)", "borderwidth": 1,
                "font": {"color": "#334155", "size": 11},
            },
        )
        st.plotly_chart(focus, width="stretch", key=f"{key_prefix}_recommendation_detail", config=MAP_PLOTLY_CONFIG)

        cells = tower_cells_lookup_all[tower_cells_lookup_all.yangon_tower_id == tower_id].copy()
        if len(cells):
            st.markdown("**Technical cell records at the nearest existing tower**")
            cell_cols = [x for x in ["radio", "Network", "MCC", "MNC", "TAC", "CID", "RANGE", "LAT", "LON"] if x in cells.columns]
            shown = cells[cell_cols].copy().rename(columns={"LAT": "cell_latitude", "LON": "cell_longitude", "RANGE": "reported_range_m"})
            st.dataframe(shown, width="stretch", hide_index=True, height=min(330, 70 + 35 * len(shown)))

    with st.expander("Model explanation for this recommendation", expanded=False):
        render_ai_assessment(assess_site(float(r['lat']), float(r['lon'])), key_prefix=f"{key_prefix}_recommend_ai", technical_only=True)
    return True


def show_map_selection(event, candidate_source, recs, key_prefix):
    tags = []
    for point in event_points(event):
        cd = _mapping_get(point, "customdata", []) or []
        try:
            cd = list(cd)
        except TypeError:
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
            "Local areas": len(subset),
            "Population 2020": round(subset.population_2020.sum()),
            "Median tower distance (km)": round(float(subset.nearest_tower_km.median()), 2),
            "Largest tower distance (km)": round(float(subset.nearest_tower_km.max()), 2),
            "Underserved local areas": int(underserved.sum()),
            "Population in underserved areas": round(subset.loc[underserved, "population_2020"].sum()),
        })
    return pd.DataFrame(rows)


# ---------- sidebar ----------
inject_global_styles()
st.sidebar.markdown(
    """
    <div class="gv-sidebar-brand">
        <div class="gv-sidebar-mark">GV</div>
        <div>
            <div class="gv-sidebar-title">Scenario settings</div>
            <div class="gv-sidebar-subtitle">Area and assumptions</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Set the region and planning assumptions. The defaults are ready for a quick briefing.")

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

scenario_area = "All project areas" if len(selected_areas) == len(ANALYSIS_AREAS) else ", ".join(selected_areas)
scenario_method = "GeoVision AI" if recommendation_engine.startswith("GeoVision AI") else "Planning rules"
st.sidebar.markdown(
    f"""
    <div class="gv-scenario-card">
        <div class="gv-scenario-label">Current scenario</div>
        <div class="gv-scenario-summary">
            <strong>{scenario_area}</strong><br>
            {service_radius_km:.1f} km service radius · gap ≥ {threshold_km:.1f} km<br>
            {n_sites} candidate sites · {scenario_method}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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
brand_logo_path = BASE / "assets" / "university_logo.jpg"
brand_logo_b64 = image_to_base64(brand_logo_path)
render_university_header(brand_logo_b64)
render_hero(selected_areas, len(towers))
st.markdown(
    """
    <details class="gv-scope">
        <summary>Planning scope &amp; data notes</summary>
        <p>
            This is a decision-support tool, not a live operator network monitor. Disaster Impact AI can use
            external hazard sources, while population and telecom coverage remain geographic estimates that
            require field surveys and RF engineering validation.
        </p>
    </details>
    """,
    unsafe_allow_html=True,
)
render_kpi_cards(
    baseline_metrics,
    len(towers),
    int((candidates.nearest_tower_km >= threshold_km).sum()),
    service_radius_km,
)

# ---------- tabs ----------
t_overview, t_population, t_gap, t_recommend, t_ai, t_disaster, t_rain, t_method = st.tabs(
    [
        "Overview",
        "Tower load",
        "Coverage gaps",
        "Suggested sites",
        "Site checker",
        "Hazard analysis",
        "Rainfall",
        "Methodology",
    ]
)

with t_overview:
    st.subheader("Where are the current coverage gaps?")
    fig = base_map(selected_areas, towers, zoom=8.45 if len(selected_areas) > 1 else 9.1)
    add_towers(fig, towers)
    add_candidates(fig, candidates, "Underserved local areas")
    add_recommendations(fig, recs)
    st.caption("Orange/red points show local areas farther from existing towers. Green numbered points show suggested locations for further field review.")
    overview_event = st.plotly_chart(fig, width="stretch", key="overview_gap_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
    overview_detail = st.container()
    with overview_detail:
        show_map_selection(overview_event, candidates, recs, "overview")

    st.markdown("#### Coverage need by area")
    st.dataframe(summary_area, width="stretch", hide_index=True)

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
            "Population 2020": round(pop_total),
            "Mapped tower sites": len(area_towers),
            "Average population / site": round(pop_total / max(len(area_towers), 1)),
            "Median estimated nearby population": round(area_towers.estimated_population_nearest.median()),
            "Highest estimated nearby population": round(area_towers.estimated_population_nearest.max()),
            "Tower sites in historic flood areas": int((area_towers.flood_frequency > 0).sum()),
        })
    st.dataframe(pd.DataFrame(p_rows), width="stretch", hide_index=True)

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
        st.plotly_chart(fig, width="stretch", config=MAP_PLOTLY_CONFIG)

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
        st.dataframe(top, width="stretch", hide_index=True, height=520)

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
        under_event = st.plotly_chart(fig, width="stretch", key="underserved_gap_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
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
        st.dataframe(top, width="stretch", hide_index=True, height=490)

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
    rec_event = st.plotly_chart(fig, width="stretch", key="recommendation_map", on_select="rerun", selection_mode="points", config=MAP_PLOTLY_CONFIG)
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
    if "expected_population_coverage" in simple:
        simple["Expected Population Coverage"] = simple["expected_population_coverage"].round(0).astype(int)
    else:
        simple["Expected Population Coverage"] = simple["population_2020"].round(0).astype(int)
    simple["Recommendation Reason"] = simple.get(
        "recommendation_reason", simple.apply(_recommendation_reason, axis=1)
    )
    simple = simple[[
        "rank", "adm3_name", "adm4_name", "Nearest Tower (km)",
        "Population Need", "Disaster Risk", "Overall Site Score", "Expected Population Coverage",
        "Recommendation", "Recommendation Reason",
    ]].rename(columns={
        "rank": "Rank",
        "adm3_name": "Township",
        "adm4_name": "Ward / Village Tract",
    })
    st.markdown("#### Priority list")
    st.dataframe(simple, width="stretch", hide_index=True)

    export_cols = [
        "rank", "analysis_area", "adm3_name", "adm4_name", "lat", "lon", "population_2020",
        "nearest_tower_id", "nearest_tower_lat", "nearest_tower_lon", "nearest_tower_km",
        "gap_score", "population_score", "elevation_m", "elevation_score", "hazard_score", "safety_score", "suitability_score",
    ]
    for optional_col in [
        "tower_location", "priority_score", "expected_population_coverage",
        "recommendation_reason", "ai_probability", "ai_decision", "recommendation_engine",
    ]:
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
        st.dataframe(technical_table, width="stretch", hide_index=True)

    st.download_button(
        "Download suggested locations with coordinates (CSV)",
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
    st.subheader("Hazard impact analysis")
    st.caption(
        "Screen tower exposure to flood, cyclone, earthquake, or compound hazards. "
        "Results are planning scores for prioritization, not public warning forecasts."
    )

    hazard_label_to_type = {
        "Flood / Heavy Rain": "flood",
        "Cyclone": "cyclone",
        "Earthquake": "earthquake",
        "Compound Risk": "compound",
    }
    feature_contracts = {
        "flood": "GEE GSMaP rainfall • SRTM terrain • historic flood susceptibility; Dynamic World is displayed as environmental context",
        "earthquake": "USGS post-event magnitude/depth/distance signal • historic seismic exposure • isolation and tower vulnerability",
        "cyclone": "JTWC current position and forecast track/wind • packaged terrain and historical exposure • tower vulnerability",
        "compound": "Flood, Earthquake and Cyclone model outputs • site isolation",
    }

    stored_hazard_type = st.session_state.get("hazard_ai_selected_type", "flood")
    default_hazard_label = next(
        (label for label, kind in hazard_label_to_type.items() if kind == stored_hazard_type),
        "Flood / Heavy Rain",
    )
    if "hazard_model_selector" not in st.session_state:
        st.session_state["hazard_model_selector"] = default_hazard_label
    hazard_label = st.radio(
        "Hazard model",
        list(hazard_label_to_type.keys()),
        horizontal=True,
        key="hazard_model_selector",
    )
    selected_hazard_type = hazard_label_to_type[hazard_label]
    st.session_state["hazard_ai_selected_type"] = selected_hazard_type
    hz_status = hazard_model_status(selected_hazard_type)

    with st.expander("Data and model details", expanded=False):
        st.caption(f"{hz_status['model_type']} · {hz_status['training_towers']:,} training towers · {hz_status['training_rows']:,} proxy/synthetic rows")
        st.caption(
            f"**{hazard_label} data:** {feature_contracts[selected_hazard_type]}. "
            "Scores are planning exposure/impact scores, not calibrated physical disaster probabilities."
        )
        st.caption("Inputs used by the trained model: " + ", ".join(name.replace("_", " ") for name in hz_status["features"]))
        st.caption("Training rows include repeated towers and synthetic scenarios. Validation measures fit to proxy labels; it does not measure observed disaster forecasting accuracy.")
        if selected_hazard_type == "cyclone":
            st.caption("The historical proxy contains one cyclone record. IBTrACS retraining and live GEE rainfall/land-cover inputs are not included in this model.")

    h1, h2 = st.columns([1.4, 1])
    with h1:
        hazard_mode_label = st.radio(
            "Data source",
            ["Automatic", "Live only", "Demo data"],
            horizontal=True,
            key="hazard_data_source_selector",
            help=(
                "Live Flood uses Google Earth Engine GSMaP, SRTM and Dynamic World. Live Cyclone uses keyless JTWC operational forecast products. "
                "Live Earthquake uses the USGS earthquake catalog because Earth Engine is not a real-time seismic-event source. "
                "Automatic keeps the analysis available by using cached project data when a live source cannot be reached."
            ),
        )
    with h2:
        st.caption("Automatic may use historical fallback. Live only refuses fallback. Demo data does not describe current hazard conditions.")

    gee_project_id = None
    gee_service_json = None
    try:
        gee_section = st.secrets.get("gee", {})
        gee_project_id = gee_section.get("project_id")
        gee_service_json = gee_section.get("service_account_json")
    except StreamlitSecretNotFoundError:
        gee_section = {}

    mode_map = {
        "Automatic": "auto",
        "Live only": "gee",
        "Demo data": "local",
    }
    analysis_signature = (selected_hazard_type, tuple(sorted(selected_areas)), mode_map[hazard_mode_label])

    selected_hazard_towers = tower_sites_all[tower_sites_all.analysis_area.isin(selected_areas)].copy()
    run_col, info_col = st.columns([0.8, 2.2])
    with run_col:
        run_hazard = st.button("Run analysis", type="primary", key="run_hazard_ai")
    with info_col:
        st.caption(f"Will run {hazard_label} AI for {len(selected_hazard_towers):,} existing tower sites in the selected analysis area(s).")

    if run_hazard:
        with st.spinner(f"Running {hazard_label} AI and preparing tower exposure..."):
            try:
                hazard_result, hazard_run = get_hazard_engine().run(
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
                st.session_state["hazard_ai_signature"] = analysis_signature
                st.session_state["hazard_ai_completed_at"] = hazard_run.get("result_context", {}).get("completed_at") or pd.Timestamp.now(tz="UTC").isoformat()
            except Exception as exc:  # noqa: BLE001 - UI boundary must surface connector/model failures
                st.session_state.pop("hazard_ai_result", None)
                st.session_state.pop("hazard_ai_run", None)
                st.session_state.pop("hazard_ai_type", None)
                st.session_state.pop("hazard_ai_signature", None)
                error_text = str(exc)
                if selected_hazard_type == "cyclone" and "JTWC" in error_text:
                    st.error("JTWC is temporarily unavailable. Use Automatic mode or try again later.")
                elif selected_hazard_type in {"flood", "earthquake", "compound"} and any(source in error_text for source in ("GEE", "USGS", "GSMaP", "JTWC")):
                    st.error("The live hazard source is temporarily unavailable. Use Automatic mode or try again later.")
                else:
                    st.error("The analysis could not be completed. Review the technical details below.")
                with st.expander("Technical details", expanded=False):
                    st.code(f"{type(exc).__name__}: {error_text}")

    hazard_result = st.session_state.get("hazard_ai_result")
    hazard_run = st.session_state.get("hazard_ai_run", {})
    hazard_areas = st.session_state.get("hazard_ai_areas")
    hazard_saved_type = st.session_state.get("hazard_ai_type")

    current_result = (
        hazard_result is not None
        and hazard_areas == tuple(selected_areas)
        and hazard_saved_type == selected_hazard_type
        and st.session_state.get("hazard_ai_signature") == analysis_signature
    )
    if current_result:
        render_hazard_results(
            hazard_result, hazard_run, hz_status,
            hazard_label=hazard_label, hazard_type=selected_hazard_type,
            requested_mode=mode_map[hazard_mode_label], selected_areas=selected_areas,
            service_radius_km=service_radius_km, all_towers=tower_sites_all,
            pop_grid=pop_grid, area_id=AREA_ID, simulate_coverage=simulate_population_coverage,
            base_map=base_map, add_track=add_cyclone_forecast_track,
            map_config=MAP_PLOTLY_CONFIG, land_cover_classes=DYNAMIC_WORLD_CLASSES,
            completed_at=st.session_state.get("hazard_ai_completed_at"),
        )
    else:
        if hazard_result is not None:
            st.warning("Analysis settings changed. Run analysis again to refresh the results for the selected model, area and data source.")
        st.markdown(
            '<section class="gv-empty-state"><h4>Your analysis will appear here</h4>'
            '<p>Choose a hazard and data source, then run the analysis. You will get an exposure map, '
            'a tower-review shortlist and an explicit what-if coverage scenario.</p>'
            '<small>Demo data works without credentials. Live Flood requires Earth Engine setup.</small></section>',
            unsafe_allow_html=True,
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
    st.plotly_chart(fig, width="stretch")

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
    st.dataframe(latest, width="stretch", hide_index=True)

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
    st.plotly_chart(fig, width="stretch", config=MAP_PLOTLY_CONFIG)

with t_method:
    st.subheader("Data inventory and methodology")
    st.markdown(
        "**Multi-model GeoVision AI:** Model 1 (`models/tower_site_xgb.json`) recommends new tower locations. "
        "Model 2 is now a multi-hazard Disaster Impact AI with four XGBoost modules: Flood/Heavy Rain (`hazard_flood_xgb.json`), "
        "Earthquake (`hazard_earthquake_xgb.json`), Cyclone (`hazard_cyclone_xgb.json`) and Compound (`hazard_compound_xgb.json`). "
        "Flood uses live GEE GSMaP/SRTM with Dynamic World environmental context when available; Cyclone uses keyless JTWC operational current/forecast products; Earthquake uses recent USGS events because GEE is not the appropriate real-time earthquake-event source. "
        "All four are hackathon/MVP exposure models with pseudo-label limitations and should not be presented as calibrated physical disaster probabilities."
    )
    drows = pd.DataFrame([
        ["WorldPop population GeoTIFF", meta.get("population_grid_points", 0), "Usable", "Population count per raster pixel; reference year 2020"],
        ["Yangon elevation GeoTIFF", meta.get("elevation_candidate_count", 0), "Usable", f"Elevation sampled at candidate points; {meta.get('elevation_candidate_min_m', 0):.0f}-{meta.get('elevation_candidate_max_m', 0):.0f} m in current candidate set"],
        ["Yangon admin boundaries", meta.get("yangon_admin3_count", 0), "Usable", f"Valid-on {meta.get('boundary_valid_on', '')}"],
        ["Yangon tower cells", meta.get("yangon_tower_cell_count", 0), "Usable with caveat", "Coordinate-dedup creates tower-site proxies"],
        ["Historic flood polygons", meta.get("flood_feature_count", 0), "Usable", "Flood footprint/frequency inside analysis region"],
        ["Rainfall time series", meta.get("rainfall_rows_yangon", 0), "Usable", f"{meta.get('rainfall_date_min')} to {meta.get('rainfall_date_max')}"],
        ["Earthquakes near Yangon", meta.get("earthquake_rows_used", 0), "Usable", "Historical exposure index"],
        ["Cyclone labels", meta.get("cyclone_rows_used", 0), "Limited", "Only one uploaded cyclone record"],
    ], columns=["Layer", "Rows/pixels/features", "Status", "Use"])
    st.dataframe(drows, width="stretch", hide_index=True)

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
        - **Flood / Heavy Rain AI:** live GEE mode validates JAXA GSMaP rainfall and combines it with SRTM terrain and historic flood susceptibility. Dynamic World land cover is displayed as environmental context and does not enter the trained model.
        - **Earthquake AI:** live mode queries recent USGS earthquake events, converts magnitude/depth/distance into a tower-level event signal, and combines it with the project's historic seismic exposure and tower vulnerability proxies.
        - **Cyclone AI:** live mode queries fresh JTWC operational current/forecast products through the U.S. Naval Research Laboratory ATCF feed, applies forecast-lead decay, and derives a tower-level track/wind signal. The model then combines that signal with historic cyclone exposure, terrain/flood context and tower isolation. No API key is required.
        - **Historical cyclone limitation:** the packaged model still uses the project's limited historical exposure proxy. IBTrACS is the recommended archive for the next historical feature/retraining pipeline; it has not yet been completed in this package.
        - **Compound AI:** a trained meta-model combines the Flood AI, Earthquake AI and Cyclone AI scores plus tower isolation to estimate multi-hazard planning exposure.
        - Earthquake, cyclone and compound targets are transparent pseudo-labels. Replace them with verified event/outage labels before making operational probability claims.

        **4. AI-driven disaster coverage impact**
        - The selected AI hazard score replaces the old manual severity scenario risk.
        - Towers above the selected AI exposure threshold are treated as unavailable for planning sensitivity analysis.
        - Population is re-routed to the nearest surviving tower. The ten cached neighbours accelerate lookup; a spatial search finds surviving alternatives beyond that cache when necessary.
        - If no surviving alternative is inside the planning service radius, that population is counted as potentially losing coverage.
        """
    )

    st.markdown("#### Critical interpretation")
    st.error(
        "Estimated population per site is NOT the number of real customers connected to that tower. Actual users require operator subscriber/traffic data. "
        "Likewise, the planning radius is NOT RF propagation. A production model should add antenna frequency, height, azimuth, transmit power, terrain/DEM, buildings, handover/traffic logs, backup power and tower outage history."
    )
