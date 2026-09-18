# GeoVision AI v11.1 — Bandwidth Map Contrast

## Why the map was almost entirely red

The old color scale was linear from 0 to the 10 Mbps target. With the bundled full-project defaults (automatic technology assignment, GSM/UMTS/LTE budgets 0.2/10/100 Mbps per site proxy, 5% active population, 5 km radius and 500 m display cells), most calculated values are much smaller than that target. They occupied a tiny portion of the gradient.

A diagnostic on 10 September 2026 found 6,796 fully supported display cells and 4,107 unsupported/mixed cells. Among fully supported cells, 97.88% of means were below 1 Mbps; the median was about 0.031 Mbps and the 90th percentile about 0.279 Mbps. Separately, 99.84% of eligible baseline population was below the 10 Mbps target. Display-cell counts/quantiles are not population-weighted; the population shortfall statistic is. These are results of the packaged scenario assumptions, not measurements of Myanmar service quality.

The assumptions remain unvalidated as documented in [BANDWIDTH_TECHNOLOGY_ASSUMPTIONS.md](BANDWIDTH_TECHNOLOGY_ASSUMPTIONS.md). Changing their values just to produce more green would not establish better accuracy. This release changes the display only.

## New color-scale control

- **Speed differences (log)** is the default planning view. Low values are separated across a softer red/amber/yellow palette. Yellow remains below target; it is not a pass/adequacy indicator. Green is reserved for display-cell means meeting or exceeding the target.
- **Target comparison (linear)** restores the original mapping: red at zero, yellow at half the target, green at the target or above. Small values can still look uniformly red here; that is consistent with the scenario output.
- The legend displays actual Mbps ticks, not normalized color coordinates. At a 10 Mbps target, the detail ticks are 0, 0.01, 0.1, 1 and 10+ Mbps. Their nonlinear spacing is explicitly labelled.
- Raw means and ranges are preserved in hover with three significant digits, so small values no longer appear merely as `0.0 Mbps`. The below-target population summary is shown above the map to remain visible even when the palette looks less severe.
- Unknown/partially supported locations remain gray. Green refers to a cell's supported-population-weighted mean, not a guarantee for every location or resident in it. Hover still contains the range and below-target population share.
- The chosen scale persists across technology changes, measured/planning mode changes and the round trip to Overview. Measured-download rendering and its linear scale are unchanged.

## Exact color mapping and invariants

For target `T` and valid mean `v < T`, the detail color coordinate is:

```text
0.97 × log1p(10000 × v / T) / log1p(10000)
```

At or above `T`, it is 1.0. All below-target values lie at or below 0.97; the green palette band starts at 0.985. Thus a below-target cell cannot be promoted to green by ranking it against weaker cells. The same value and target give the same color regardless of viewport, other cells, technology filter or study-area selection. Zero remains zero. Above-target values share the endpoint color but retain their uncapped raw Mbps in hover. Normalized color coordinates are never displayed as Mbps.

The simulation, site budgets, active-user assumptions, nearest-site assignments, full-project demand, technology inventory, cell polygons, population summaries and AI model files are unchanged. No smoothing/interpolation, percentile auto-ranging or fabricated speed readings are introduced. This is a readability improvement, not an improvement in observed network performance or validated model accuracy.

## Verification

The automated suite passes **355 tests**, including 18 new color-scale cases: monotonic mapping, zero/target endpoints, green-threshold separation, correct legend units, identical raw hover values and masks across scales, viewport-independent colors, invalid inputs and all-unknown areas. Scoped Ruff checks pass. The suite's nine existing earthquake/rainfall timedelta deprecation warnings remain unrelated to this update.

The isolated Streamlit test passed on the actual project grid, comparing raw chart values and the population summary between log/linear modes and checking persistent controls across technology/evidence switches. The synthetic measured-upload test also passed unchanged. The full Streamlit smoke test passed in the user's Anaconda environment with a simulated Rasterio block, including the bandwidth-to-overview round trip, site checker and all four local hazard workspaces. These are programmatic UI checks; no browser screenshot/pixel review or live-feed certification is claimed.

## Run

Stop the old Streamlit process with Ctrl+C, extract the complete v11.1 ZIP into a new folder, and run `python -m streamlit run app.py` from the folder containing `app.py`. In Bandwidth evidence, leave **Color scale → Speed differences (log)** selected for the detail view. No new dependency or security-setting change is needed. Existing releases and private credentials are not modified; no real secrets are included in the ZIP.

Runtime change: `ui/bandwidth_evidence_map.py`. Updated tests/scripts: `tests/test_bandwidth_colors.py`, `scripts/bandwidth_smoke_test.py`, `scripts/app_smoke_test.py`. The previous technology-based presets, operator-free bandwidth controls, original basemap/header and optional Rasterio startup fix are retained.
