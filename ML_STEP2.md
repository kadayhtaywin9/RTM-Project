# ML Step 2 — Train the XGBoost tower-site prototype

## What Step 2 does

This project now contains a trained XGBoost binary classifier:

`models/tower_site_xgb.json`

It predicts the probability that a candidate site belongs to the current
`optimal_site = 1` class.

## Features

The model expects these columns in this exact order:

1. `gap_score`
2. `population_score`
3. `is_rural`
4. `safety_score`
5. `elevation_score`

## Training data

- Rows: 194
- Positive (`optimal_site=1`): 49
- Negative (`optimal_site=0`): 145
- Township spatial groups: 36

## Spatial validation

The script uses 4-fold `StratifiedGroupKFold` with township as the group.
A township is never simultaneously in the train and test portions of the same
fold.

Out-of-fold results:

- ROC-AUC: 0.9895
- Average Precision: 0.9614
- Accuracy at 0.50: 0.9639
- Precision at 0.50: 0.9200
- Recall at 0.50: 0.9388
- F1 at 0.50: 0.9293

Provisional threshold selected from out-of-fold predictions:

- Threshold: 0.65
- Precision: 0.9574
- Recall: 0.9184
- F1: 0.9375

The threshold is provisional and should be revisited when real operator labels
are available.

## Re-train the model

From the project directory:

```bash
python ml/train_xgboost.py
```

## Generated artifacts

- `models/tower_site_xgb.json` — deployable XGBoost model
- `models/model_metadata.json` — feature order, threshold and limitations
- `models/xgboost_metrics.json` — pooled and per-fold metrics
- `data/xgboost_cv_metrics.csv` — per-fold metrics
- `data/xgboost_oof_predictions.csv` — spatial out-of-fold predictions
- `data/xgboost_feature_importance.csv` — gain-based importance
- `data/xgboost_threshold_scan.csv` — threshold/precision/recall/F1 scan

## Critical limitation

The Step-1 target is a pseudo-label derived from the existing rule-based
planning score. Therefore the strong Step-2 metrics mainly show that XGBoost can
learn and generalize the current planning rule across the available township
groups. They do **not** prove that the model predicts real tower deployment
success.

The next deployment step is to build the exact same five input features for any
latitude/longitude clicked in Streamlit, load `tower_site_xgb.json`, call
`predict_proba`, and create the site-assessment report.
