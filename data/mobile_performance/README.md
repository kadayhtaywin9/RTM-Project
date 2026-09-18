# Ookla mobile source data

Source: **Speedtest by Ookla Global Fixed and Mobile Network Performance Maps**, collected 10 September 2026.

Documentation: https://github.com/teamookla/ookla-open-data

License: **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International**.
https://creativecommons.org/licenses/by-nc-sa/4.0/

Keep this attribution and license notice when sharing extracts. Follow the noncommercial and share-alike terms. Source-specific conditions apply to this dataset; no commercial rights were obtained.

The combined files contain 650 quarterly mobile tile records for 2026 Q1/Q2, over 479 distinct tiles, summarizing 2,498 tests. Each record is a tile-quarter aggregate, not one test, person, tower or radio sector. Speeds are source means; Mbps = original kbps / 1,000. Missing operator, radio generation and serving-cell fields are intentionally blank.

Modifications: geographic/column selection of the source Parquet; original numeric attributes retained; tile WKT/GeoJSON decoded from original zoom-16 quadkeys; explicit unit conversion, source/quarter metadata and single-test/boundary flags. Tile geometry is not clipped, and speeds are not interpolated. Reference: https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system.

The CSV is the dashboard's validated source. Parquet and GeoJSON provide equivalent modeling/GIS formats. Per-quarter source Parquet files preserve selected original attributes without the redundant global WKT column. Their source URLs, object versions, hashes and limitations are in the matching manifests. `dataset.json` records the CSV integrity and period contract.

301 records have only one test. Device counts are not additive unique-device counts across tiles/quarters. No numerical confidence intervals, radio-specific speeds, tower capacities or current coverage can be inferred from these aggregates alone. See `../../MOBILE_MEASUREMENTS.md` for integration and modeling limitations.
