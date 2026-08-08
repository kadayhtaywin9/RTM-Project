from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

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
AREA_NAMES = list(AREA_TOWNSHIPS)
TOWNSHIP_TO_AREA = {t: a for a, towns in AREA_TOWNSHIPS.items() for t in towns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build population-load and flood/rainfall features for the GeoAI telecom dashboard.")
    parser.add_argument("--population-tif", required=True)
    parser.add_argument("--rainfall-csv", required=True)
    parser.add_argument("--flood-geojson", required=True)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--k-neighbors", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    admin3 = gpd.read_file(data_dir / "yangon_admin3.geojson")
    admin4 = gpd.read_file(data_dir / "analysis_admin4.geojson").reset_index(drop=True)
    towers_all = pd.read_csv(data_dir / "tower_sites.csv")
    candidates = pd.read_csv(data_dir / "candidate_village_tracts.csv")
    flood = gpd.read_file(args.flood_geojson).to_crs(4326)
    rainfall = pd.read_csv(args.rainfall_csv)

    selected_townships = [t for towns in AREA_TOWNSHIPS.values() for t in towns]
    admin3_sel = admin3[admin3.adm3_name.isin(selected_townships)].copy()
    analysis_geom = admin3_sel.geometry.union_all()

    # ---------- population raster clip ----------
    with rasterio.open(args.population_tif) as src:
        if src.crs is None:
            raise ValueError("Population GeoTIFF has no CRS.")
        if str(src.crs) != "EPSG:4326":
            raise ValueError(f"Expected EPSG:4326 WorldPop raster, got {src.crs}.")
        win = from_bounds(*analysis_geom.bounds, transform=src.transform).round_offsets().round_lengths()
        tform = src.window_transform(win)
        pop_arr = src.read(1, window=win, masked=True)
        inside = geometry_mask([analysis_geom.__geo_interface__], out_shape=pop_arr.shape, transform=tform, invert=True)
        valid = inside & (~pop_arr.mask) & np.isfinite(pop_arr.data) & (pop_arr.data > 0)

    rows, cols = np.where(valid)
    lon = tform.c + (cols + 0.5) * tform.a + (rows + 0.5) * tform.b
    lat = tform.f + (cols + 0.5) * tform.d + (rows + 0.5) * tform.e
    population = pop_arr.data[valid].astype(np.float32)

    # Analysis-area raster ID (1..4).
    area_shapes = []
    for i, area in enumerate(AREA_NAMES, start=1):
        g = admin3_sel[admin3_sel.adm3_name.isin(AREA_TOWNSHIPS[area])].geometry.union_all()
        area_shapes.append((g, i))
    area_r = rasterize(area_shapes, out_shape=pop_arr.shape, transform=tform, fill=0, dtype="uint8")
    area_id = area_r[valid].astype(np.uint8)

    # Admin-4 population sums.
    a4_shapes = [(g, i + 1) for i, g in enumerate(admin4.geometry)]
    a4_r = rasterize(a4_shapes, out_shape=pop_arr.shape, transform=tform, fill=0, dtype="uint16")
    a4_id = a4_r[valid]
    pop_by_a4 = np.bincount(a4_id, weights=population, minlength=len(admin4) + 1)[1:]
    admin4["population_2020"] = pop_by_a4
    admin4[["adm4_pcode", "adm3_name", "adm4_name", "population_2020"]].to_csv(
        data_dir / "admin4_population_2020.csv", index=False
    )

    # Attach admin-4 population to candidate points by point-in-polygon, which also resolves duplicate 'Urban' names.
    cand_gdf = gpd.GeoDataFrame(
        candidates.reset_index(names="candidate_id"),
        geometry=gpd.points_from_xy(candidates.lon, candidates.lat),
        crs=4326,
    )
    joined = gpd.sjoin(
        cand_gdf,
        admin4[["adm4_pcode", "population_2020", "geometry"]],
        predicate="within",
        how="left",
    )
    # If a representative point lands on a boundary, use nearest admin-4 polygon within the same township.
    pop_map = joined.drop_duplicates("candidate_id").set_index("candidate_id")[["adm4_pcode", "population_2020"]]
    candidates = candidates.reset_index(names="candidate_id").merge(pop_map, left_on="candidate_id", right_index=True, how="left")
    missing = candidates.population_2020.isna()
    if missing.any():
        # Fallback: use nearest polygon centroid in the same township.
        cent = admin4.copy()
        cent["centroid"] = cent.geometry.representative_point()
        for idx, row in candidates[missing].iterrows():
            sub = cent[cent.adm3_name == row.adm3_name]
            if len(sub):
                d = sub.centroid.distance(gpd.GeoSeries([gpd.points_from_xy([row.lon], [row.lat])[0]], crs=4326).iloc[0])
                best = sub.iloc[int(np.argmin(d.to_numpy()))]
                candidates.loc[idx, "adm4_pcode"] = best.adm4_pcode
                candidates.loc[idx, "population_2020"] = best.population_2020
    candidates["population_2020"] = candidates.population_2020.fillna(0.0)
    max_pop = max(float(candidates.population_2020.max()), 1.0)
    candidates["population_score"] = np.log1p(candidates.population_2020) / np.log1p(max_pop)
    candidates.drop(columns=["candidate_id"], errors="ignore").to_csv(data_dir / "candidate_village_tracts.csv", index=False)

    # ---------- tower population catchments ----------
    towers = towers_all[towers_all.adm3_name.isin(selected_townships)].copy().reset_index(drop=True)
    towers.insert(0, "tower_id", np.arange(len(towers), dtype=np.int32))
    towers["analysis_area"] = towers.adm3_name.map(TOWNSHIP_TO_AREA)
    adm2_map = admin3_sel.drop_duplicates("adm3_name").set_index("adm3_name")[["adm2_name", "adm2_pcode"]]
    towers = towers.merge(adm2_map, left_on="adm3_name", right_index=True, how="left")

    transformer = Transformer.from_crs(4326, 32647, always_xy=True)
    px, py = transformer.transform(lon, lat)
    tx, ty = transformer.transform(towers.lon.to_numpy(), towers.lat.to_numpy())
    tree = cKDTree(np.column_stack([tx, ty]))
    k = max(2, int(args.k_neighbors))
    dist_m, nearest_idx = tree.query(np.column_stack([px, py]), k=k)
    nearest_dist_km = (dist_m / 1000.0).astype(np.float32)
    nearest_idx = nearest_idx.astype(np.int32)

    primary_idx = nearest_idx[:, 0]
    primary_dist = nearest_dist_km[:, 0]
    pop_nearest = np.bincount(primary_idx, weights=population, minlength=len(towers))
    pop_primary_5km = np.bincount(
        primary_idx[primary_dist <= 5.0],
        weights=population[primary_dist <= 5.0],
        minlength=len(towers),
    )
    towers["estimated_population_nearest"] = pop_nearest
    towers["estimated_population_primary_5km"] = pop_primary_5km
    towers["estimated_people_per_cell"] = towers.estimated_population_nearest / towers.cell_count.clip(lower=1)

    # ---------- historic flood exposure ----------
    flood_sel = flood[flood.intersects(analysis_geom)].copy()
    if len(flood_sel):
        flood_sel.to_file(data_dir / "historic_flood_analysis.geojson", driver="GeoJSON")
        shapes = [(r.geometry, int(r.flood_frequency)) for _, r in flood_sel.sort_values("flood_frequency").iterrows()]
        flood_r = rasterize(shapes, out_shape=pop_arr.shape, transform=tform, fill=0, dtype="uint8")
        flood_frequency_pixel = flood_r[valid].astype(np.uint8)

        tower_points = gpd.GeoDataFrame(
            towers[["tower_id", "lon", "lat"]],
            geometry=gpd.points_from_xy(towers.lon, towers.lat),
            crs=4326,
        )
        sj = gpd.sjoin(tower_points, flood_sel[["flood_frequency", "count", "geometry"]], predicate="intersects", how="left")
        texp = sj.groupby("tower_id").agg(flood_frequency=("flood_frequency", "max"), flood_count=("count", "max"))
        towers = towers.merge(texp, left_on="tower_id", right_index=True, how="left")
        towers[["flood_frequency", "flood_count"]] = towers[["flood_frequency", "flood_count"]].fillna(0)
        max_freq = max(float(flood_sel.flood_frequency.max()), 1.0)
        towers["flood_history_score"] = np.log1p(towers.flood_frequency) / np.log1p(max_freq)
        flood_pop_by_tower = np.bincount(
            primary_idx[flood_frequency_pixel > 0],
            weights=population[flood_frequency_pixel > 0],
            minlength=len(towers),
        )
        towers["historical_flood_exposed_population_assigned"] = flood_pop_by_tower
    else:
        flood_frequency_pixel = np.zeros(len(population), dtype=np.uint8)
        towers["flood_frequency"] = 0
        towers["flood_count"] = 0
        towers["flood_history_score"] = 0.0
        towers["historical_flood_exposed_population_assigned"] = 0.0
        (data_dir / "historic_flood_analysis.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    towers.to_csv(data_dir / "tower_sites_population.csv", index=False)

    # Compact population service grid; the dashboard can simulate failures without the 343 MB source raster.
    np.savez_compressed(
        data_dir / "population_service_grid.npz",
        lon=lon.astype(np.float32),
        lat=lat.astype(np.float32),
        population=population,
        area_id=area_id,
        flood_frequency=flood_frequency_pixel,
        nearest_tower_idx=nearest_idx,
        nearest_tower_dist_km=nearest_dist_km,
    )

    # ---------- rainfall time series ----------
    rainfall["date"] = pd.to_datetime(rainfall["date"])
    rain_yangon = rainfall[(rainfall.adm_level == 2) & rainfall.PCODE.astype(str).isin(admin3_sel.adm2_pcode.unique())].copy()
    pcode_name = admin3_sel.drop_duplicates("adm2_pcode").set_index("adm2_pcode")["adm2_name"].to_dict()
    rain_yangon["adm2_name"] = rain_yangon.PCODE.map(pcode_name)
    # Percentile rank within each Yangon division over the supplied 5-year series.
    for col in ["rfh", "r1h", "r3h"]:
        rain_yangon[f"{col}_percentile"] = rain_yangon.groupby("PCODE")[col].rank(pct=True, method="average")
    rain_yangon.sort_values(["date", "PCODE"]).to_csv(data_dir / "yangon_rainfall_5y.csv", index=False, date_format="%Y-%m-%d")

    # ---------- metadata ----------
    meta_path = data_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    area_population = {AREA_NAMES[i - 1]: float(population[area_id == i].sum()) for i in range(1, len(AREA_NAMES) + 1)}
    meta.update({
        "population_source": Path(args.population_tif).name,
        "population_reference_year": 2020,
        "analysis_population_total": float(population.sum()),
        "analysis_area_population_2020": area_population,
        "population_grid_points": int(len(population)),
        "population_nearest_neighbors_precomputed": int(k),
        "flood_feature_count": int(len(flood_sel)),
        "flood_tower_intersections": int((towers.flood_frequency > 0).sum()),
        "historical_flood_exposed_population_2020": float(population[flood_frequency_pixel > 0].sum()),
        "rainfall_rows_yangon": int(len(rain_yangon)),
        "rainfall_date_min": rain_yangon.date.min().strftime("%Y-%m-%d") if len(rain_yangon) else None,
        "rainfall_date_max": rain_yangon.date.max().strftime("%Y-%m-%d") if len(rain_yangon) else None,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Summary for validation / presentation.
    tower_summary = towers.groupby("analysis_area", as_index=False).agg(
        tower_sites=("tower_id", "count"),
        assigned_population_2020=("estimated_population_nearest", "sum"),
        mean_people_per_site=("estimated_population_nearest", "mean"),
        median_people_per_site=("estimated_population_nearest", "median"),
        max_people_per_site=("estimated_population_nearest", "max"),
        flood_exposed_sites=("flood_frequency", lambda s: int((s > 0).sum())),
    )
    tower_summary.to_csv(data_dir / "population_tower_summary.csv", index=False)

    print("Built population + disaster features")
    print(json.dumps(meta, indent=2))
    print(tower_summary.to_string(index=False))


if __name__ == "__main__":
    main()
