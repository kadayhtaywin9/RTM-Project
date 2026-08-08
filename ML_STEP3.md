# ML Step 3 — Deploy XGBoost inside the Streamlit dashboard

Step 3 connects the saved Step-2 model to the actual GeoAI application.

## What changed

### 1. XGBoost is now a recommendation engine

The sidebar contains:

- `XGBoost AI (trained model)` — default
- `Rule-based baseline` — preserves the previous weighted planning formula

In XGBoost mode, every available candidate row is scored with:

`models/tower_site_xgb.json`

The model probability is converted to a 0–100 suitability score. The existing
minimum-spacing rule is then applied so the highest-scoring recommendations do
not cluster together.

### 2. AI Site Checker

A new `🤖 AI Site Checker` tab lets the user select an arbitrary coordinate.
With `streamlit-folium` installed, the user can click directly on the map.
An exact latitude/longitude input is also available as a fallback.

The model returns:

- XGBoost suitability probability
- `OPTIMAL CANDIDATE` / `NOT OPTIMAL`
- provisional decision threshold
- nearest observed tower and distance
- Admin-4 population
- elevation
- the five normalized XGBoost feature values
- feature-level prediction drivers
- downloadable text assessment report

### 3. Existing map markers also invoke XGBoost

Clicking an orange/red gap point or a green recommendation marker now shows the
normal GIS/tower details followed by the XGBoost AI site assessment.

## Runtime feature construction

The Step-2 model expects these five features in this exact order:

1. `gap_score`
2. `population_score`
3. `is_rural`
4. `safety_score`
5. `elevation_score`

For an arbitrary map click:

- **Coverage gap:** calculated at the exact clicked coordinate using the nearest
  observed tower-site proxy, then normalized with the same 25 km scale used by
  the candidate data.
- **Population demand:** uses WorldPop 2020 population for the containing Admin-4
  polygon and the same log normalization as training.
- **Rural priority:** uses the containing Admin-4 planning classification.
- **Hazard safety:** uses the containing Admin-4 hazard-safety proxy used during
  training.
- **Elevation advantage:** samples `data/yangon_elevation.tif` at the exact
  clicked coordinate and applies the same candidate-set min/max normalization.

This design deliberately distinguishes exact-coordinate features from
Admin-4-level proxy features in the report.

## XGBoost explanation

The dashboard uses the native XGBoost `pred_contribs=True` output to display
per-feature margin contributions. The UI reports whether each factor pushes the
prediction toward or away from the optimal class and its share of absolute
contribution magnitude.

These contribution shares are explanation aids. They are not causal effects or
probability percentage points.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The new click map requires the additional dependencies already listed in
`requirements.txt`:

```text
folium
streamlit-folium
```

If `streamlit-folium` cannot be imported, the dashboard still runs and the AI
Site Checker falls back to exact latitude/longitude entry.

## Validation performed in Step 3

The arbitrary-coordinate runtime feature extractor was tested on all 194
original candidate coordinates.

- Feature/prediction runtime checks succeeded for 194 / 194 sites.
- XGBoost probability difference versus direct candidate-row prediction: 0.0
  across the checked set.
- Optimal/not-optimal decision agreement at the 0.65 threshold: 100%.
- Outside-analysis coordinates are rejected rather than extrapolated.

## Critical limitation

The model remains an MVP/hackathon model because the target is a pseudo-label
created from the previous planning rule. It is now technically integrated and
usable in the project, but it is not yet operator-validated deployment
intelligence.

Before operational telecom site selection, add real labels/KPIs plus RF,
capacity, land, access, power, backhaul, permitting and engineering feasibility
features.
