# Tower Site XGBoost Prototype — Model Card

**Purpose:** rank/evaluate Yangon candidate telecom tower sites using the five
features already present in the GeoAI dashboard.

**Target:** `optimal_site` (0/1)

**Training rows:** 194

**Positive / negative:** 49 / 145

**Spatial validation:** 4-fold township-aware stratified group CV.

**OOF ROC-AUC:** 0.9895

**OOF Average Precision:** 0.9614

**Provisional decision threshold:** 0.65

**Model file:** `tower_site_xgb.json`

## Warning

This is a hackathon/MVP model. Its target is generated from the previous
rule-based site suitability score, so it currently learns that planning logic.
It is not yet trained on operator-confirmed successful/failed deployments.
