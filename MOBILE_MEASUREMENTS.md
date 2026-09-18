# Collected mobile measurements

## Source and license

Source: **Speedtest by Ookla Global Fixed and Mobile Network Performance Maps**, mobile layer, collected 10 September 2026.

- Documentation: https://github.com/teamookla/ookla-open-data
- Catalogue: https://registry.opendata.aws/speedtest-global-performance/
- License: **CC BY-NC-SA 4.0**, https://creativecommons.org/licenses/by-nc-sa/4.0/

Retain attribution and follow the noncommercial/share-alike terms when using or sharing this dataset. No additional commercial rights or source endorsement were obtained. This source-specific license notice concerns the included measurements; it does not assert a new license for unrelated project code.

Exact URLs, ETags, access dates, selected columns and source snapshot hashes are recorded in `data/mobile_performance/*_manifest.json`. `dataset.json` pins the dashboard CSV checksum and quarter/count contract. An intentional dataset refresh requires new validation and registry values, not disabling the integrity check.

## Meaning and calculations

| Quarter | Tile records | Tests | Single-test records | Test-weighted mean download |
| --- | ---: | ---: | ---: | ---: |
| 2026 Q1 | 319 | 1,363 | 139 | 34.737 Mbps |
| 2026 Q2 | 331 | 1,135 | 162 | 34.976 Mbps |

These full-extract summaries include whole tiles intersecting the original project boundary. Their means describe the sampled participating tests, not all residents, subscribers, networks or times. Means are reconstructed from rounded source tile averages weighted by source test counts. Device counts are unique within a tile-quarter, not across the dataset. 171 tiles appear in both quarters, and repeat observations are correlated.

Each row is one mobile tile in one calendar quarter. The start is inclusive and end exclusive. The dataset contains **means**, not raw tests, medians, tower capacities or calibrated forecasts. Added Mbps values are original kbps / 1,000. Unknown metadata remains blank; real zero readings are retained.

The approximately 600-metre source grid is kept at zoom 16. Tile polygons are reconstructed from original quadkeys using standard Web Mercator equations, checked against original centroids and the collected WKT. Geometry reference: https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system. Source attributes are unchanged. No population-pixel upsampling, clipping, synthetic speed generation or interpolation is performed in the collected-measurements view.

v14 adds a separate **Estimated full AOI** view that uses the original records as spatial examples. It generates labelled estimates across selected AOI polygons, with source-distance and extrapolation flags. Generated values never become extra source observations or training labels. The default spatial holdout error is worse than a simple mean baseline; this is an illustrative fill, not a validated speed-coverage model. See [full-AOI method and evaluation](FULL_AOI_ESTIMATES.md).

## Dashboard behavior

- A fresh bandwidth view opens the latest bundled quarter, Q2 2026, with a minimum of three tests per tile. Previously selected evidence modes and filters are preserved within the session.
- Only one quarter is mapped. The study-area filter selects whole intersecting tiles; some observations can lie outside the boundary. The UI counts displayed tiles whose centroids lie outside the selection.
- Red, yellow and green encode source mean download relative to an editable comparison target. Original values above the target remain available in hover. Changing a target never changes source speeds or summary means.
- Gray tiles fall below the selected minimum-test filter. They are excluded from speed summaries and downloads. Uncolored areas have no displayed observation; they are not zero Mbps or proof of absent service.
- Selecting a minimum of one test exposes every available tile and a single-test warning. Three or more tests do not establish representativeness or confidence.
- Downloads contain only the displayed records, source-grid GeoJSON, selection metadata and attribution/limitations. They preserve distinct tile-quarter identifiers and missing operator/radio/cell fields.
- The quarterly table in the modeling expander uses the same area and test-count filters. Differences between quarterly averages are not controlled estimates of network improvement.

## Modeling scope

The dataset is now available to the project through `engine.mobile_performance.load_collected_measurements()` and as typed Parquet/CSV. The runtime reader uses the CSV and existing pandas/Shapely dependencies, with no GEE authentication, network request or rasterio import. The registry/checksum, row uniqueness, dates, units, counts and tile geometry are validated before any display.

It supplies candidate targets for a **tile-quarter aggregate mobile-throughput model**. It does not label individual towers or identify the technology used by a test. No existing site/hazard AI model has been retrained, and the capacity-sharing scenario is deliberately not automatically fitted. Many combinations of active-user counts and site capacities could explain the same throughput; fitting their ratio alone does not identify either physical input.

Before training a defensible model:

1. Define the target and prediction date. Join only contextual features available at that date; the historical tower inventory does not prove current site status.
2. Keep each tile-quarter as one record. Do not replicate an average across towers/pixels and count the copies as independent labels.
3. Compare simple baselines with any learned model, using geographic block and later-quarter holdouts. Avoid neighboring-tile leakage and examine repeat tiles separately.
4. Do not assume same-quarter test counts, uploads or measured speeds are available for an unsampled prediction location. Avoid target-derived features.
5. Report independent prediction error, sample support and out-of-distribution areas. The aggregates contain neither raw readings nor within-tile variance, so test counts alone cannot yield valid confidence intervals.
6. Obtain operator/radio/serving-cell measurements and current RF, spectrum, traffic and backhaul inputs before claiming technology-specific user speeds or tower capacities.

This evidence is a useful starting point, not proof that an accurate citywide model now exists.
