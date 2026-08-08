# GeoAI Population, Connectivity & Disaster-Resilient Telecom Dashboard

Analysis areas: **Yangon City (33 YCDC townships), Hmawbi, Thanlyin and Kyauktan**.

This version upgrades the original connectivity-gap dashboard with three new user datasets:

1. `mmr_ppp_2020_UNadj whole population.tif` — WorldPop 2020 population raster.
2. `mmr-rainfall-subnat-5ytd.csv` — WFP/CHIRPS dekadal rainfall indicators, supplied period 2022-01-01 to 2026-07-21.
3. `historic_flood.geojson` — historical flood polygons and flood-frequency attributes.

The large 343 MB source population GeoTIFF is **not duplicated in this project ZIP**. It has already been preprocessed into compact dashboard files. The included `preprocess_population_disaster.py` can regenerate the derived files from the raw sources.


## Clickable map upgrade in this package

This package intentionally keeps the **older Yangon dashboard layout and analysis scope** while adding the newer map interaction style.

- Click an **orange/red Admin-4 gap point** to see the exact nearest observed tower-site proxy, the gap-point and tower coordinates, a line connecting them, network/radio information, and the individual cell rows grouped at that site.
- Click a **green numbered recommendation** to see the complete recommendation row, exact recommended latitude/longitude, suitability score, current gap, nearest existing tower, and the nearest tower's underlying cell records.
- The **Tower recommendations** tab includes a CSV download containing the current recommendations, candidate latitude/longitude, nearest-tower ID/coordinates, gap distance, population and suitability inputs.
- Nearest-tower lookup uses the full **8,540 observed Yangon tower-site proxies**, because a gap point near the edge of Hmawbi/Thanlyin/Kyauktan can legitimately have its nearest observed site in a neighboring Yangon township. The population/disaster simulation remains the original four-area Yangon model.

## What the dashboard now answers

### 1. How many people are associated with one tower?

The source telecom file contains **cell observations**, not verified physical tower inventory. Cells at the same coordinates were deduplicated into **observed tower-site proxies**.

For the population estimate:

1. The WorldPop raster is clipped to the four analysis areas.
2. Every populated raster pixel is assigned to its nearest observed tower-site proxy.
3. The raster population value is added to that site's geographic catchment.
4. The dashboard reports both:
   - **Nearest-catchment population**: all people for whom the site is nearest, regardless of distance.
   - **Primary population within 5 km**: a more conservative planning-load measure.

This is **not subscriber count**. Actual tower users require operator traffic, SIM, handover or sector-utilization data.

### 2. How does disaster affect coverage?

For every population raster pixel, the preprocessing stage stores its **10 nearest tower-site alternatives**.

During a disaster scenario:

1. Each site receives a comparative scenario-risk score.
2. Sites above the selected risk threshold are assumed unavailable.
3. Population pixels whose primary site fails are reassigned to their nearest surviving alternative.
4. A person is counted as covered only when the surviving alternative is within the selected planning service radius.
5. The dashboard reports:
   - primary population affected;
   - population rerouted to backup sites;
   - population losing coverage;
   - baseline vs post-disaster coverage percentage;
   - surviving sites absorbing the largest extra load.

The service radius is a **planning assumption**, not RF propagation modeling.

## New population results from the supplied WorldPop raster

Approximate 2020 population inside the analysis polygons:

| Analysis area | Population 2020 | Observed tower-site proxies | Simple population/site |
|---|---:|---:|---:|
| Yangon City | 6,385,122 | 7,759 | ~823 |
| Hmawbi | 302,941 | 114 | ~2,657 |
| Thanlyin | 327,663 | 136 | ~2,409 |
| Kyauktan | 202,182 | 20 | ~10,109 |
| **Total** | **7,217,908** | **8,029** | — |

Because tower catchments can cross administrative boundaries, the tower-level nearest-catchment assignment does not exactly equal each administrative population row when grouped by the tower's township.

The median nearest-catchment loads in the current calculation are approximately:

- Yangon City: **367 people/site**
- Hmawbi: **1,648 people/site**
- Thanlyin: **885 people/site**
- Kyauktan: **4,230 people/site**

Kyauktan remains the clearest infrastructure-pressure area: it has only 20 observed tower-site proxies in the analysis dataset and much larger geographic gaps.

## Flood + rainfall integration

The new historical flood file contains **8,507 polygons nationally/regionally in the supplied file**, of which **1,162 intersect the four analysis areas**.

Within the analysis region:

- **41 observed tower-site proxies** fall inside a historical flood polygon.
- About **312,516 people** in the 2020 raster fall inside the historical flood footprint.

The rainfall file contains 10-day (`rfh`), one-month rolling (`r1h`) and three-month rolling (`r3h`) rainfall, their long-term averages, and anomaly percentages. For the flood/heavy-rain scenario, the dashboard uses the **one-month rainfall percentile within each Yangon ADM2 division** as the event-stress indicator.

Flood/heavy-rain hazard score:

`0.70 × historic flood susceptibility + 0.30 × rainfall stress`

Historic flood susceptibility comes from the flood-frequency attribute at the tower location.

## Example disaster result

With the default demonstration settings:

- Planning service radius: **5 km**
- Scenario: **Flood / Heavy Rain**
- Severity: **3 / 5**
- Rainfall snapshot: **2026-07-21**
- Site-unavailability threshold: **risk >= 0.60**

Across all four areas, the scenario currently estimates roughly:

- **6 tower-site proxies assumed unavailable**
- **38,379 people** whose primary site is affected
- **14,921 people** rerouted successfully
- **23,458 people** losing modeled coverage
- baseline coverage **97.05%** -> post-disaster coverage **96.72%**

These values are scenario outputs, not calibrated failure predictions.

## Population-aware tower recommendation

The recommendation engine is now weighted by:

- 45% coverage gap
- 30% population demand
- 15% rural priority
- 10% hazard safety

Weights remain adjustable in the dashboard.

Population demand is the WorldPop total inside each Admin-4 area, log-normalized so very large urban units do not completely dominate the model.

## Files

- `app.py` — Streamlit dashboard.
- `geoai_engine.py` — suitability scoring, disaster risk and population re-routing logic.
- `preprocess_population_disaster.py` — rebuild population/flood/rainfall features from the raw GeoTIFF/CSV/GeoJSON.
- `data/tower_sites_population.csv` — tower-site proxies with estimated population load and flood exposure.
- `data/population_service_grid.npz` — compact population pixels + ten nearest tower alternatives.
- `data/admin4_population_2020.csv` — population totals per Admin-4 polygon.
- `data/yangon_rainfall_5y.csv` — filtered Yangon ADM2 rainfall time series with rainfall percentiles.
- `data/historic_flood_analysis.geojson` — flood polygons intersecting the analysis areas.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows:

```text
run_windows.bat
```

macOS/Linux:

```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

## Rebuild from the original source datasets

```bash
python preprocess_population_disaster.py \
  --population-tif "/path/to/mmr_ppp_2020_UNadj whole population.tif" \
  --rainfall-csv "/path/to/mmr-rainfall-subnat-5ytd.csv" \
  --flood-geojson "/path/to/historic_flood.geojson"
```

## Critical limitations

This is a **strategic GeoAI screening and resilience-planning tool**, not a radio-network engineering certification.

The following data would be required to estimate real tower users and physical coverage accurately:

- operator subscriber and traffic counters;
- antenna height, azimuth, tilt and transmit power;
- carrier frequency and bandwidth;
- tower sector geometry;
- terrain/elevation and building obstruction;
- actual outage/failure history;
- backup batteries/generators and grid reliability;
- backhaul/fiber/microwave dependency;
- tower capacity and congestion limits.

The cyclone dataset still contains only one uploaded cyclone record, so cyclone outputs should be treated as experimental/limited.


## Elevation-aware tower recommendation

The dashboard now samples `data/yangon_elevation.tif` at every Admin-4 candidate coordinate.
Each candidate has `elevation_m` and an `elevation_score` normalized from 0 to 1 across the Yangon candidate set.
Higher elevation increases tower suitability. The default recommendation weights are:

- 40% coverage gap
- 25% population demand
- 10% rural priority
- 10% hazard safety
- 15% elevation advantage

Elevation is a planning advantage proxy, not a complete RF model. A final site survey should also check slope/access, antenna height, line-of-sight, land availability, power and backhaul.

## XGBoost AI site assessment — Step 3

The trained Step-2 XGBoost model is now deployed in the dashboard.

- The default **Recommendation engine** is `XGBoost AI (trained model)`.
- The original weighted recommendation logic remains available as
  `Rule-based baseline`.
- The new **🤖 AI Site Checker** tab can evaluate an arbitrary map coordinate
  and return an optimal/not-optimal model decision plus a downloadable report.
- Existing gap and recommendation markers also show the XGBoost assessment when
  clicked.

Runtime inference is implemented in `site_ai.py`. Exact coordinate tower gap and
elevation are combined with containing Admin-4 population/rural/safety features
to form the same five model inputs used in training.

See `ML_STEP3.md` for the full deployment and validation notes.
