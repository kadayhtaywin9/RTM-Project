"""Runtime XGBoost site assessment for the Yangon GeoAI dashboard.

The model is a Step-2 hackathon/MVP classifier trained on pseudo-labels generated
from the previous planning rule. This module keeps runtime feature construction
consistent with those training features for an arbitrary latitude/longitude.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Point
import xgboost as xgb
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"

FEATURES = [
    "gap_score",
    "population_score",
    "is_rural",
    "safety_score",
    "elevation_score",
]

FEATURE_LABELS = {
    "gap_score": "Coverage gap",
    "population_score": "Population demand",
    "is_rural": "Rural priority",
    "safety_score": "Hazard safety",
    "elevation_score": "Elevation advantage",
}

EARTH_RADIUS_KM = 6371.0088
GAP_NORMALIZATION_KM = 25.0  # matches the candidate feature construction used by the project


@lru_cache(maxsize=1)
def _assets() -> dict[str, Any]:
    candidates = pd.read_csv(DATA / "candidate_village_tracts.csv")
    towers = pd.read_csv(DATA / "tower_sites_yangon_all.csv")
    admin4 = gpd.read_file(DATA / "analysis_admin4.geojson").to_crs(4326)
    pop = pd.read_csv(DATA / "admin4_population_2020.csv")

    model = XGBClassifier()
    model.load_model(MODELS / "tower_site_xgb.json")
    metadata = json.loads((MODELS / "model_metadata.json").read_text(encoding="utf-8"))

    # Fast lookups for area-level features that were part of the training table.
    candidate_by_pcode: dict[str, pd.DataFrame] = {
        str(k): v.copy() for k, v in candidates.dropna(subset=["adm4_pcode"]).groupby("adm4_pcode")
    }
    pop_by_pcode = pop.dropna(subset=["adm4_pcode"]).drop_duplicates("adm4_pcode").set_index("adm4_pcode")

    return {
        "candidates": candidates,
        "towers": towers,
        "admin4": admin4,
        "population": pop,
        "candidate_by_pcode": candidate_by_pcode,
        "pop_by_pcode": pop_by_pcode,
        "model": model,
        "metadata": metadata,
    }


def model_status() -> dict[str, Any]:
    a = _assets()
    meta = a["metadata"]
    return {
        "model_name": meta.get("model_name", "XGBoost tower-site model"),
        "training_rows": int(meta.get("training_rows", 0)),
        "features": list(meta.get("features_in_order", FEATURES)),
        "threshold": float(meta.get("provisional_decision_threshold", 0.65)),
        "limitations": list(meta.get("limitations", [])),
    }


def _haversine_vector_km(lat: float, lon: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1 = np.radians(float(lat))
    p2 = np.radians(lat2.astype(float))
    dphi = p2 - p1
    dlambda = np.radians(lon2.astype(float) - float(lon))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _nearest_tower(lat: float, lon: float, towers: pd.DataFrame) -> tuple[pd.Series, float]:
    dist = _haversine_vector_km(
        lat,
        lon,
        towers["lat"].to_numpy(float),
        towers["lon"].to_numpy(float),
    )
    idx = int(np.argmin(dist))
    return towers.iloc[idx], float(dist[idx])


def _containing_admin4(lat: float, lon: float, admin4: gpd.GeoDataFrame) -> pd.Series | None:
    point = Point(float(lon), float(lat))
    # covers() includes points exactly on polygon boundaries.
    matches = admin4[admin4.geometry.covers(point)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _nearest_candidate_row(lat: float, lon: float, candidates: pd.DataFrame, township: str | None = None) -> pd.Series:
    subset = candidates
    if township:
        local = candidates[candidates["adm3_name"].astype(str) == str(township)]
        if not local.empty:
            subset = local
    dist = _haversine_vector_km(
        lat,
        lon,
        subset["lat"].to_numpy(float),
        subset["lon"].to_numpy(float),
    )
    return subset.iloc[int(np.argmin(dist))]


def _area_proxy_candidate(lat: float, lon: float, adm4_row: pd.Series, assets: dict[str, Any]) -> pd.Series:
    pcode = str(adm4_row.get("adm4_pcode", ""))
    group = assets["candidate_by_pcode"].get(pcode)
    if group is not None and not group.empty:
        if len(group) == 1:
            return group.iloc[0]
        return _nearest_candidate_row(lat, lon, group)
    return _nearest_candidate_row(lat, lon, assets["candidates"], str(adm4_row.get("adm3_name", "")))


def _sample_elevation(lat: float, lon: float) -> float | None:
    path = DATA / "yangon_elevation.tif"
    try:
        with rasterio.open(path) as src:
            x, y = float(lon), float(lat)
            if src.crs and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(x, y)
            val = float(next(src.sample([(x, y)]))[0])
            if src.nodata is not None and math.isclose(val, float(src.nodata), rel_tol=0.0, abs_tol=1e-9):
                return None
            if not np.isfinite(val):
                return None
            return val
    except Exception:
        return None


def extract_site_features(lat: float, lon: float) -> dict[str, Any]:
    """Build the five Step-2 model features for one arbitrary map coordinate."""
    lat = float(lat)
    lon = float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"ok": False, "reason": "Latitude/longitude are outside valid coordinate ranges."}

    assets = _assets()
    admin4_row = _containing_admin4(lat, lon, assets["admin4"])
    if admin4_row is None:
        return {
            "ok": False,
            "reason": "The selected point is outside the project's Yangon City + Hmawbi + Thanlyin + Kyauktan Admin-4 analysis boundary.",
            "lat": lat,
            "lon": lon,
        }

    proxy = _area_proxy_candidate(lat, lon, admin4_row, assets)
    nearest_tower, nearest_tower_km = _nearest_tower(lat, lon, assets["towers"])

    pcode = str(admin4_row.get("adm4_pcode", ""))
    pop_row = None
    try:
        pop_row = assets["pop_by_pcode"].loc[pcode]
    except Exception:
        pop_row = None
    if pop_row is not None:
        population_2020 = float(pop_row.get("population_2020", proxy.get("population_2020", 0.0)))
    else:
        population_2020 = float(proxy.get("population_2020", 0.0))

    max_pop = max(float(assets["candidates"]["population_2020"].max()), 1.0)
    population_score = float(np.clip(np.log1p(max(population_2020, 0.0)) / np.log1p(max_pop), 0.0, 1.0))

    elevation_m = _sample_elevation(lat, lon)
    elevation_source = "exact GeoTIFF sample"
    if elevation_m is None:
        elevation_m = float(proxy.get("elevation_m", np.nan))
        elevation_source = "Admin-4 candidate proxy (raster sample unavailable)"
    elev_min = float(assets["candidates"]["elevation_m"].min())
    elev_max = float(assets["candidates"]["elevation_m"].max())
    if np.isfinite(elevation_m) and elev_max > elev_min:
        elevation_score = float(np.clip((elevation_m - elev_min) / (elev_max - elev_min), 0.0, 1.0))
    else:
        elevation_score = float(np.clip(proxy.get("elevation_score", 0.0), 0.0, 1.0))

    feature_values = {
        "gap_score": float(np.clip(nearest_tower_km / GAP_NORMALIZATION_KM, 0.0, 1.0)),
        "population_score": population_score,
        "is_rural": float(np.clip(proxy.get("is_rural", 0.0), 0.0, 1.0)),
        "safety_score": float(np.clip(proxy.get("safety_score", 0.0), 0.0, 1.0)),
        "elevation_score": elevation_score,
    }

    return {
        "ok": True,
        "lat": lat,
        "lon": lon,
        "adm3_name": str(admin4_row.get("adm3_name", "")),
        "adm4_name": str(admin4_row.get("adm4_name", "")),
        "adm4_pcode": pcode,
        "population_2020": population_2020,
        "nearest_tower_km": nearest_tower_km,
        "nearest_tower_id": int(nearest_tower.get("yangon_tower_id", -1)),
        "nearest_tower_lat": float(nearest_tower.get("lat", np.nan)),
        "nearest_tower_lon": float(nearest_tower.get("lon", np.nan)),
        "nearest_tower_networks": str(nearest_tower.get("networks", "")),
        "nearest_tower_radios": str(nearest_tower.get("radios", "")),
        "nearest_tower_cell_count": int(nearest_tower.get("cell_count", 0)),
        "elevation_m": float(elevation_m) if np.isfinite(elevation_m) else None,
        "elevation_source": elevation_source,
        "earthquake_score_proxy": float(proxy.get("earthquake_score", np.nan)),
        "cyclone_score_proxy": float(proxy.get("cyclone_score", np.nan)),
        "flood_score_proxy": float(proxy.get("flood_score", np.nan)),
        "hazard_score_proxy": float(proxy.get("hazard_score", np.nan)),
        "feature_values": feature_values,
        "feature_sources": {
            "gap_score": "Exact clicked coordinate → nearest observed tower-site distance, normalized to 25 km",
            "population_score": "Containing Admin-4 WorldPop 2020 total",
            "is_rural": "Containing Admin-4 planning classification",
            "safety_score": "Containing Admin-4 hazard-safety proxy",
            "elevation_score": f"Clicked coordinate elevation; {elevation_source}",
        },
    }



def recommend_sites_xgb(candidates: pd.DataFrame, n_sites: int = 10, min_spacing_km: float = 8.0) -> pd.DataFrame:
    """Rank candidate rows with the trained XGBoost model and enforce spacing."""
    if candidates.empty:
        return candidates.copy()
    assets = _assets()
    model: XGBClassifier = assets["model"]
    threshold = float(assets["metadata"].get("provisional_decision_threshold", 0.65))

    df = candidates.copy()
    X = df[FEATURES].astype(float)
    prob = model.predict_proba(X)[:, 1]
    df["ai_probability"] = prob
    df["ai_decision"] = np.where(prob >= threshold, "OPTIMAL CANDIDATE", "NOT OPTIMAL")
    # Preserve the existing dashboard column contract: suitability_score is 0..100.
    df["suitability_score"] = 100.0 * prob
    df["recommendation_engine"] = "XGBoost AI"
    ranked = df.sort_values(["ai_probability", "nearest_tower_km"], ascending=False)

    selected = []
    for _, row in ranked.iterrows():
        if all(
            float(_haversine_vector_km(
                float(row.lat), float(row.lon),
                np.asarray([float(prev.lat)]), np.asarray([float(prev.lon)])
            )[0]) >= float(min_spacing_km)
            for prev in selected
        ):
            selected.append(row)
        if len(selected) >= int(n_sites):
            break

    if not selected:
        return ranked.head(0).copy()
    out = pd.DataFrame(selected).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def assess_site(lat: float, lon: float) -> dict[str, Any]:
    extracted = extract_site_features(lat, lon)
    if not extracted.get("ok"):
        return extracted

    assets = _assets()
    model: XGBClassifier = assets["model"]
    metadata = assets["metadata"]
    threshold = float(metadata.get("provisional_decision_threshold", 0.65))

    row = pd.DataFrame([[extracted["feature_values"][f] for f in FEATURES]], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    is_optimal = bool(probability >= threshold)

    # TreeSHAP-style margin contributions from the native XGBoost booster.
    dmat = xgb.DMatrix(row, feature_names=FEATURES)
    contrib = model.get_booster().predict(dmat, pred_contribs=True)[0]
    feature_contrib = contrib[:-1]
    abs_total = max(float(np.abs(feature_contrib).sum()), 1e-12)
    explanations = []
    for feature, margin_value in zip(FEATURES, feature_contrib):
        explanations.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": float(extracted["feature_values"][feature]),
            "margin_contribution": float(margin_value),
            "direction": "Toward optimal" if margin_value > 0 else "Away from optimal" if margin_value < 0 else "Neutral",
            "relative_impact_pct": 100.0 * abs(float(margin_value)) / abs_total,
        })
    explanations.sort(key=lambda x: abs(x["margin_contribution"]), reverse=True)

    extracted.update({
        "probability": probability,
        "score_100": probability * 100.0,
        "decision_threshold": threshold,
        "is_optimal": is_optimal,
        "decision": "OPTIMAL CANDIDATE" if is_optimal else "NOT OPTIMAL",
        "explanations": explanations,
        "model_name": metadata.get("model_name", "XGBoost tower-site model"),
        "model_warning": (
            "Prototype model: trained on pseudo-labels derived from the prior planning rule, not operator-confirmed deployment success."
        ),
    })
    return extracted
