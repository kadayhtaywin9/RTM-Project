"""Core GeoAI scoring utilities for the Yangon telecom resilience dashboard."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def recommend_sites(
    candidates: pd.DataFrame,
    n_sites: int = 10,
    min_spacing_km: float = 8.0,
    gap_weight: float = 0.45,
    population_weight: float = 0.30,
    rural_weight: float = 0.15,
    safety_weight: float = 0.10,
) -> pd.DataFrame:
    """Rank admin-4 candidate points and greedily enforce site diversity.

    This is an interpretable multi-criteria GeoAI suitability model, not a learned ML model.
    """
    df = candidates.copy()
    weights = np.array([gap_weight, population_weight, rural_weight, safety_weight], dtype=float)
    weights = weights / max(weights.sum(), 1e-9)
    df["suitability_score"] = 100 * (
        weights[0] * df["gap_score"].clip(0, 1)
        + weights[1] * df.get("population_score", 0).clip(0, 1)
        + weights[2] * df["is_rural"].clip(0, 1)
        + weights[3] * df["safety_score"].clip(0, 1)
    )
    ranked = df.sort_values(["suitability_score", "nearest_tower_km"], ascending=False)

    selected = []
    for _, row in ranked.iterrows():
        if all(
            haversine_km(row.lat, row.lon, prev.lat, prev.lon) >= min_spacing_km
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


def attach_rainfall_to_towers(
    towers: pd.DataFrame,
    rainfall: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """Attach WFP/CHIRPS rainfall metrics for one dekad to each tower via ADM2 PCode."""
    df = towers.copy()
    d = pd.Timestamp(date)
    rain = rainfall.copy()
    rain["date"] = pd.to_datetime(rain["date"])
    snap = rain[rain.date == d].copy()
    keep = ["PCODE", "rfh", "rfh_avg", "rfq", "r1h", "r1h_avg", "r1q", "r3h", "r3h_avg", "r3q", "r1h_percentile"]
    snap = snap[[c for c in keep if c in snap.columns]].drop_duplicates("PCODE")
    df = df.merge(snap, left_on="adm2_pcode", right_on="PCODE", how="left")
    df["rainfall_stress_score"] = df.get("r1h_percentile", pd.Series(0.0, index=df.index)).fillna(0.0).clip(0, 1)
    # Flood/heavy-rain hazard: historical flood susceptibility is the main factor; rainfall is the event trigger.
    df["flood_rain_hazard_score"] = (
        0.70 * df.get("flood_history_score", 0).fillna(0).clip(0, 1)
        + 0.30 * df["rainfall_stress_score"]
    ).clip(0, 1)
    return df


def outage_risk(
    towers: pd.DataFrame,
    scenario: str = "Flood / Heavy Rain",
    severity: int = 3,
) -> pd.DataFrame:
    """Comparative site-unavailability risk for scenario analysis.

    The score is NOT a calibrated physical failure probability because historical tower-outage
    labels and engineering vulnerability data are not available.
    """
    df = towers.copy()
    sev = float(np.clip(severity, 1, 5)) / 5.0
    scenario_lower = scenario.lower()

    earthquake = df.get("earthquake_score", pd.Series(0.0, index=df.index)).to_numpy(float)
    cyclone = df.get("cyclone_score", pd.Series(0.0, index=df.index)).to_numpy(float)
    flood_rain = df.get("flood_rain_hazard_score", pd.Series(0.0, index=df.index)).to_numpy(float)

    if "flood" in scenario_lower or "rain" in scenario_lower:
        hazard = flood_rain
    elif scenario_lower == "earthquake":
        hazard = earthquake
    elif scenario_lower == "cyclone":
        hazard = cyclone
    else:  # compound
        hazard = np.maximum.reduce([earthquake, cyclone, flood_rain])

    isolation = df.get("isolation_score", pd.Series(0.0, index=df.index)).to_numpy(float)
    legacy = df.get("legacy_score", pd.Series(0.0, index=df.index)).to_numpy(float)

    # Comparative logistic response: hazard dominates; severity scales event stress;
    # isolation and legacy technology are secondary vulnerability proxies.
    logit = -3.45 + 2.8 * hazard + 1.75 * sev + 0.9 * isolation + 0.65 * legacy
    risk = 1.0 / (1.0 + np.exp(-logit))
    df["scenario_hazard_score"] = np.clip(hazard, 0, 1)
    df["scenario_risk"] = np.clip(risk, 0, 1)
    df["risk_class"] = pd.cut(
        df["scenario_risk"],
        bins=[-0.01, 0.25, 0.50, 0.70, 1.01],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)
    return df


def simulate_population_coverage(
    grid: dict[str, np.ndarray],
    towers: pd.DataFrame,
    selected_area_ids: list[int],
    selected_areas: list[str],
    service_radius_km: float,
    failure_risk_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate population coverage before/after removing high-risk tower-site proxies.

    Population pixels are assigned to the nearest surviving tower among the precomputed K nearest
    alternatives. A pixel is considered covered when its nearest available tower is within the
    user-selected planning radius. This is a screening model, not RF propagation modeling.
    """
    pop = grid["population"].astype(float)
    area_id = grid["area_id"]
    nearest_idx = grid["nearest_tower_idx"]
    nearest_dist = grid["nearest_tower_dist_km"].astype(float)

    pixel_mask = np.isin(area_id, np.asarray(selected_area_ids, dtype=area_id.dtype))
    if not pixel_mask.any():
        return {}, towers.head(0).copy()

    n_towers = len(towers)
    risk_arr = np.zeros(n_towers, dtype=float)
    area_arr = np.empty(n_towers, dtype=object)
    risk_arr[towers.tower_id.to_numpy(int)] = towers.scenario_risk.to_numpy(float)
    area_arr[towers.tower_id.to_numpy(int)] = towers.analysis_area.to_numpy(object)
    failed = (risk_arr >= float(failure_risk_threshold)) & np.isin(area_arr, np.asarray(selected_areas, dtype=object))

    idx = nearest_idx[pixel_mask]
    dist = nearest_dist[pixel_mask]
    p = pop[pixel_mask]

    baseline_idx = idx[:, 0]
    baseline_dist = dist[:, 0]
    baseline_covered = baseline_dist <= float(service_radius_km)

    alive_matrix = ~failed[idx]
    any_alive = alive_matrix.any(axis=1)
    first_alive_pos = np.argmax(alive_matrix, axis=1)
    row = np.arange(len(idx))
    post_idx = idx[row, first_alive_pos]
    post_dist = dist[row, first_alive_pos]
    post_dist = np.where(any_alive, post_dist, np.inf)
    post_covered = post_dist <= float(service_radius_km)

    primary_failed = failed[baseline_idx]
    rerouted = baseline_covered & primary_failed & post_covered
    lost = baseline_covered & ~post_covered
    directly_affected = baseline_covered & primary_failed

    baseline_load = np.bincount(
        baseline_idx[baseline_covered],
        weights=p[baseline_covered],
        minlength=n_towers,
    )
    post_load = np.bincount(
        post_idx[post_covered],
        weights=p[post_covered],
        minlength=n_towers,
    )

    tower_load = towers.copy()
    tower_load["failed_in_scenario"] = failed[tower_load.tower_id.to_numpy(int)]
    tower_load["baseline_people_within_radius"] = baseline_load[tower_load.tower_id.to_numpy(int)]
    tower_load["post_disaster_people_within_radius"] = post_load[tower_load.tower_id.to_numpy(int)]
    base = tower_load["baseline_people_within_radius"].to_numpy(float)
    post = tower_load["post_disaster_people_within_radius"].to_numpy(float)
    tower_load["load_change_people"] = post - base
    ratio = np.zeros_like(post, dtype=float)
    np.divide(post, base, out=ratio, where=base > 0)
    ratio[(base == 0) & (post > 0)] = np.inf
    tower_load["load_ratio"] = ratio

    total_pop = float(p.sum())
    baseline_served = float(p[baseline_covered].sum())
    post_served = float(p[post_covered].sum())
    metrics = {
        "population_total": total_pop,
        "baseline_served": baseline_served,
        "baseline_uncovered": total_pop - baseline_served,
        "post_served": post_served,
        "post_uncovered": total_pop - post_served,
        "population_directly_affected": float(p[directly_affected].sum()),
        "population_rerouted": float(p[rerouted].sum()),
        "population_losing_coverage": float(p[lost].sum()),
        "baseline_coverage_pct": 100.0 * baseline_served / max(total_pop, 1e-9),
        "post_coverage_pct": 100.0 * post_served / max(total_pop, 1e-9),
        "failed_towers": int(failed.sum()),
        "selected_failed_towers": int(((risk_arr >= failure_risk_threshold) & np.isin(area_arr, selected_areas)).sum()),
        "k_alternatives": int(nearest_idx.shape[1]),
    }
    return metrics, tower_load


def coverage_summary(candidates: pd.DataFrame, threshold_km: float = 5.0) -> pd.DataFrame:
    df = candidates.copy()
    df["underserved"] = df["nearest_tower_km"] >= threshold_km
    df["underserved_population"] = np.where(df.underserved, df.get("population_2020", 0), 0.0)
    return (
        df.groupby("adm3_name", as_index=False)
        .agg(
            village_tracts=("adm4_name", "count"),
            rural_tracts=("is_rural", "sum"),
            population_2020=("population_2020", "sum"),
            median_gap_km=("nearest_tower_km", "median"),
            max_gap_km=("nearest_tower_km", "max"),
            underserved_tracts=("underserved", "sum"),
            underserved_population=("underserved_population", "sum"),
        )
        .sort_values("median_gap_km", ascending=False)
    )
