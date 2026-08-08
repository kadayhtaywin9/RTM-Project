# XGBoost Tower-Site Model — Step 1: Training Data

This project now contains a reproducible first training dataset for the tower-site suitability model.

## Files added

- `data/tower_training_data.csv` — 194 labeled Yangon candidate sites.
- `data/tower_training_data_summary.json` — label/feature summary.
- `ml/prepare_training_data.py` — regenerates the training CSV from `data/candidate_village_tracts.csv`.

## Step-1 model features

The first XGBoost classifier will use only these five inputs:

1. `gap_score` — priority from distance/gap in the existing network.
2. `population_score` — normalized population-demand signal.
3. `is_rural` — rural-priority indicator.
4. `safety_score` — inverse of combined disaster hazard.
5. `elevation_score` — normalized elevation preference already used by the dashboard.

Do **not** train on `label_suitability_score`; it exists only to document how the bootstrap label was made.

## Target

`optimal_site` is the binary target:

- `1` = top-quartile candidate under the current planning rule.
- `0` = remaining candidate.

The threshold is calculated from the data rather than hard-coded. In this dataset the 75th-percentile cutoff is approximately **45.2123/100**, producing **49 positive** and **145 negative** examples.

## Why this is a prototype label

The current project does not contain operator ground-truth outcomes such as post-deployment traffic, coverage improvement, construction cost, ROI, or engineering approval. Therefore Step 1 uses a weakly supervised / pseudo-label derived from the dashboard's existing multi-criteria planning rule.

This is acceptable for an MVP/hackathon demonstration, but the model should later be retrained with real operator/site-performance labels if available.

## Spatial validation field

`spatial_group` is set to the candidate's Yangon township (`adm3_name`). Step 2 should use this field with grouped/spatial validation rather than a random train/test split.

## Rebuild the training data

From the project root:

```bash
python ml/prepare_training_data.py
```

Expected outputs:

```text
data/tower_training_data.csv
data/tower_training_data_summary.json
```

## Next step

Step 2 trains an `XGBClassifier` on the five features, evaluates it using held-out townships, chooses a probability threshold, and saves the trained model for Streamlit inference.
