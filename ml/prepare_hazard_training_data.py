#!/usr/bin/env python3
"""Rebuild Model-2 training data and the cached tower terrain features.

The current prototype target is deliberately pseudo-labelled because the project
does not yet include event-linked flood/no-flood ground truth at tower locations.
Use this only as an MVP scaffold; replace hazard_target_score with observed labels
when available.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

towers = pd.read_csv(DATA / "tower_sites_population.csv")
rain = pd.read_csv(DATA / "yangon_rainfall_5y.csv")
rain["date"] = pd.to_datetime(rain["date"])

with rasterio.open(DATA / "yangon_elevation.tif") as src:
    dem = src.read(1).astype("float32")
    resx, resy = src.res
    mean_lat = float(towers["lat"].mean())
    dx_m = resx * 111320 * np.cos(np.deg2rad(mean_lat))
    dy_m = resy * 111320
    dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")
    inv = ~src.transform
    xy = np.array([inv * (x, y) for x, y in zip(towers["lon"], towers["lat"])])
    cols = np.clip(np.floor(xy[:, 0]).astype(int), 0, src.width - 1)
    rows = np.clip(np.floor(xy[:, 1]).astype(int), 0, src.height - 1)
    elev = dem[rows, cols]
    slp = slope[rows, cols]

static = towers[[
    "tower_id", "adm3_name", "analysis_area", "adm2_pcode", "lat", "lon",
    "flood_history_score", "flood_frequency",
]].copy()
static["elevation_m"] = elev
static["slope_deg"] = slp
static["elevation_risk"] = 1.0 - np.clip(static["elevation_m"] / 40.0, 0, 1)
static["slope_risk"] = 1.0 - np.clip(static["slope_deg"] / 5.0, 0, 1)
static.to_csv(DATA / "tower_hazard_static_features.csv", index=False)

dates = np.array(sorted(rain["date"].unique()))
sel = pd.to_datetime(dates[np.unique(np.linspace(0, len(dates)-1, 16).round().astype(int))])
parts = []
for dt in sel:
    snap = rain[rain["date"] == dt][["PCODE", "r1h", "r1h_percentile"]].drop_duplicates("PCODE")
    d = static.merge(snap, left_on="adm2_pcode", right_on="PCODE", how="left")
    d["date"] = dt
    d["rain_30d_mm"] = d["r1h"].fillna(float(rain["r1h"].median()))
    d["rain_30d_percentile"] = d["r1h_percentile"].fillna(0.5).clip(0, 1)
    h = d["flood_history_score"].fillna(0).clip(0, 1)
    r = d["rain_30d_percentile"]
    e = d["elevation_risk"].clip(0, 1)
    s = d["slope_risk"].clip(0, 1)
    d["hazard_target_score"] = (0.30*r + 0.30*h + 0.14*e + 0.06*s + 0.20*(r*h)).clip(0, 1)
    parts.append(d)

train = pd.concat(parts, ignore_index=True)
cols = [
    "tower_id", "adm3_name", "analysis_area", "adm2_pcode", "date", "rain_30d_mm",
    "rain_30d_percentile", "flood_history_score", "elevation_risk", "slope_risk",
    "hazard_target_score",
]
train[cols].to_csv(DATA / "hazard_training_data.csv", index=False)
print(f"Wrote {len(train):,} rows to data/hazard_training_data.csv")
