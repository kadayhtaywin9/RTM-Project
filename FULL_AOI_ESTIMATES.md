# Full AOI estimates

## Open the map

Select **Overview → Show bandwidth mask → Show full AOI estimate**. The button carries your observed-data quarter and minimum-test filter into the estimate view. You can also select **Estimated full AOI** directly under Map evidence. Use the sidebar to select one or all project study areas. Return to **Collected mobile measurements** to see only the original evidence.

The new map replaces the measurement/tower overlays with one full-polygon layer. All four project study areas (Yangon City, Hmawbi, Thanlyin and Kyauktan), comprising 36 township boundaries, are supported; this does not mean every township in the wider Yangon Region. Seikgyikanaungto is included even though it has no tower records in the inventory. Islands are retained, polygon holes are respected, and areas without population pixels are included. Online OpenStreetMap tiles require internet; calculation uses local data without an API key or GEE authentication.

## What is calculated

1. Select one source quarter and retain its unique tiles meeting the minimum-test count. The initial choice is Q2 2026 with at least three tests per tile. This is a screening choice, not a guarantee of representativeness.
2. Use reference tiles from the whole bundled project, even for a smaller selected AOI. No artificial source records are added. No radio, operator, population, current network or RF conditions are inferred from the samples.
3. Build a 500 m grid in UTM zone 47N (EPSG:32647), clipped to the selected polygons. A 1000 m display option is available. Display resolution is not accuracy. Each cell receives an estimate at a representative point inside its clipped geometry; this is not an integrated cell-average prediction. Geographic reprojection has small boundary approximation differences.
4. Find up to eight nearest source tile centres and calculate `sum(weight * source_mean) / sum(weight)`, with `weight = 1 / max(distance_metres, 1)^2`. An exact source-centre query returns that source value. Every output is deterministic, bounded by source speeds and clearly labelled as generated.
5. Flag extrapolation if the estimate point is outside the source centres' convex hull OR more than 5 km from its nearest reference. These descriptive cutoffs do not establish statistical confidence. The source averages cover tiles, not exact point measurements; using their centres is itself an approximation.

The method is inverse-distance weighting, not a newly trained AI model. The eight-neighbour setting and power of two are explicit illustration choices, not optimized parameters. [Esri's IDW documentation](https://doc.esri.com/en/arcgis-pro/latest/help/analysis/geostatistical-analyst/how-inverse-distance-weighted-interpolation-works.html) explains the distance weighting and the lack of prediction standard errors. Closely situated places can have very different mobile service, so geographic smoothness is an assumption, not evidence.

## Colors and interpretation

The speed layer uses red at 0, yellow at half the chosen display maximum and green at that maximum or higher. The initial maximum is 50 Mbps. It is a display scale, not a quality threshold or promised service rate. Changing it does not alter the estimates. Hover shows the actual estimated speed, nearest-reference distance, source quarter and support flag.

**Color by distance to measurements** replaces the speed layer with a blue distance scale. More distant places have less local sample support; distance is not a calibrated uncertainty interval. The area-mean card weights generated cells by their mapped area, not by tests, subscribers or population. Do not describe it as the average user speed.

## Evaluation of the bundled default

All four project areas (36 townships), Q2 2026, at least three tests per source tile, 500 m grid:

| Check | Result |
| --- | ---: |
| Generated display cells | 11,203 |
| Filled project AOI area | 2,622.47 km² |
| Reference tiles | 106 |
| Tests represented by those references | 847 |
| Area requiring extrapolation | 81.1% |
| Area more than 5 km from a reference centre | 69.8% |
| Held-out tile-mean MAE | 20.12 Mbps |
| Held-out tile-mean RMSE | 25.12 Mbps |
| Simple training-mean baseline MAE | 17.09 Mbps |

The evaluation hides each 5 km spatial block in turn and excludes training samples within 1 km of its held-out tile centres. Predictions for 106 withheld tile averages are compared with their source values. The baseline uses only that fold's training averages. Error metrics weight held-out tiles equally; they are not per-user metrics or confidence intervals. Minimum-test filtering happens before this check, so results concern that selected sample set.

**The spatial illustration performs worse than the simple mean baseline.** This result is shown above the map, with details in an expander. The data do not support a claim that the colored local patterns are accurate. The check does not validate distant unsampled areas or current network performance. Source sampling bias, temporal differences and absent operator/radio information remain unresolved. More observations across the AOIs and an independently validated model are needed for reliable coverage inference; smoothing more pixels cannot supply that evidence.

## Downloads and reproducibility

Expand **Method and full AOI download**. The ZIP contains:

- `estimated_aoi.csv`: generated values, area, inside-polygon point, nearest-source distance, support flags, quarter and explicit `data_kind=estimated_from_quarterly_samples`.
- `estimated_aoi.geojson`: the same classification and generated values attached to the clipped cells. Some numeric fields are rounded for JSON serialization.
- `reference_tiles.csv`: the selected original example values and counts, never mixed with generated rows.
- `method_and_validation.json`: source CSV hash, resolution, threshold, method, attribution, validation and area statistics.
- `README.txt`: interpretation and licensing notes.

Reproduce the default checks from the project directory with `python scripts/aoi_bandwidth_check.py`. Add `--export-dir path/to/new-output` to create a ready-to-use full-project estimate ZIP plus holdout predictions. The exporter will not overwrite an existing estimate ZIP.

Keep the Ookla for Good / Speedtest by Ookla attribution and source links supplied in the export. The source's [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/) also governs the distributed derived dataset. No original source file or existing hazard/site model is modified by this view.
