#!/usr/bin/env python3
"""Train the Step-2 Yangon Telecom GeoAI XGBoost prototype.

Reads:
    data/tower_training_data.csv

Writes:
    models/tower_site_xgb.json
    models/model_metadata.json
    models/xgboost_metrics.json
    data/xgboost_cv_metrics.csv
    data/xgboost_oof_predictions.csv
    data/xgboost_feature_importance.csv
    data/xgboost_threshold_scan.csv

Important:
The current target is a pseudo-label created in Step 1 from the dashboard's
planning rule. High performance therefore means the model can reproduce that
planning logic; it is not proof of real-world tower deployment performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

FEATURES = [
    "gap_score",
    "population_score",
    "is_rural",
    "safety_score",
    "elevation_score",
]
TARGET = "optimal_site"
GROUP = "spatial_group"

BASE_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2,
)


def make_model(y_train: pd.Series) -> XGBClassifier:
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = negative / max(positive, 1)
    return XGBClassifier(**BASE_PARAMS, scale_pos_weight=scale_pos_weight)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_file = root / "data" / "tower_training_data.csv"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    required = FEATURES + [TARGET, GROUP]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    groups = df[GROUP].astype(str)

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(df), dtype=float)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        model = make_model(y.iloc[train_idx])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[test_idx] = prob
        y_test = y.iloc[test_idx]

        fold_rows.append({
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_positive": int(y.iloc[train_idx].sum()),
            "test_positive": int(y_test.sum()),
            "test_negative": int(len(y_test) - y_test.sum()),
            "test_groups": " | ".join(sorted(df.iloc[test_idx][GROUP].unique())),
            "accuracy_at_0_5": float(accuracy_score(y_test, pred)),
            "precision_at_0_5": float(precision_score(y_test, pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_test, pred, zero_division=0)),
            "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "average_precision": float(average_precision_score(y_test, prob)),
        })

    # Provisional probability threshold selected from out-of-fold predictions.
    threshold_rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (oof_prob >= threshold).astype(int)
        threshold_rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        })
    threshold_df = pd.DataFrame(threshold_rows)
    best_f1 = threshold_df["f1"].max()
    best = (
        threshold_df[threshold_df["f1"] == best_f1]
        .sort_values(["precision", "threshold"], ascending=[False, True])
        .iloc[0]
    )
    best_threshold = float(best["threshold"])

    pred05 = (oof_prob >= 0.5).astype(int)
    predbest = (oof_prob >= best_threshold).astype(int)

    pooled = {
        "rows": int(len(df)),
        "positive": int(y.sum()),
        "negative": int((1-y).sum()),
        "spatial_groups": int(groups.nunique()),
        "cv": "StratifiedGroupKFold",
        "n_splits": 4,
        "random_state": 42,
        "roc_auc_oof": float(roc_auc_score(y, oof_prob)),
        "average_precision_oof": float(average_precision_score(y, oof_prob)),
        "threshold_0_5": {
            "accuracy": float(accuracy_score(y, pred05)),
            "precision": float(precision_score(y, pred05, zero_division=0)),
            "recall": float(recall_score(y, pred05, zero_division=0)),
            "f1": float(f1_score(y, pred05, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred05).tolist(),
        },
        "provisional_threshold": {
            "threshold": best_threshold,
            "selection_rule": "max out-of-fold F1; precision used as tie-breaker",
            "accuracy": float(accuracy_score(y, predbest)),
            "precision": float(precision_score(y, predbest, zero_division=0)),
            "recall": float(recall_score(y, predbest, zero_division=0)),
            "f1": float(f1_score(y, predbest, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predbest).tolist(),
        },
        "folds": fold_rows,
    }

    # Final deployable model is fitted on all Step-1 rows after CV evaluation.
    final_model = make_model(y)
    final_model.fit(X, y)
    model_path = models_dir / "tower_site_xgb.json"
    final_model.save_model(model_path)

    # Verify the serialized model reloads correctly.
    reloaded = XGBClassifier()
    reloaded.load_model(model_path)
    before = final_model.predict_proba(X)[:, 1]
    after = reloaded.predict_proba(X)[:, 1]
    reload_diff = float(np.max(np.abs(before - after)))

    booster = final_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    importance = pd.DataFrame({
        "feature": FEATURES,
        "gain": [float(gain.get(feature, 0.0)) for feature in FEATURES],
    })
    total_gain = importance["gain"].sum()
    importance["gain_share"] = importance["gain"] / total_gain if total_gain else 0.0
    importance = importance.sort_values("gain", ascending=False)

    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_csv(root / "data" / "xgboost_cv_metrics.csv", index=False)

    oof = df[
        ["candidate_id", "adm3_name", "adm4_name", "lat", "lon", GROUP, TARGET]
    ].copy()
    oof["xgb_oof_probability"] = oof_prob
    oof["predicted_at_0_5"] = pred05
    oof["predicted_at_provisional_threshold"] = predbest
    oof.to_csv(root / "data" / "xgboost_oof_predictions.csv", index=False)

    importance.to_csv(root / "data" / "xgboost_feature_importance.csv", index=False)
    threshold_df.to_csv(root / "data" / "xgboost_threshold_scan.csv", index=False)

    positive = int(y.sum())
    negative = int(len(y) - positive)
    final_spw = negative / max(positive, 1)

    metadata = {
        "model_name": "Yangon Telecom GeoAI XGBoost Site Suitability Prototype",
        "model_type": "XGBClassifier",
        "target": TARGET,
        "positive_class": 1,
        "features_in_order": FEATURES,
        "spatial_group": GROUP,
        "training_rows": int(len(df)),
        "training_positive": positive,
        "training_negative": negative,
        "class_weight_scale_pos_weight": final_spw,
        "provisional_decision_threshold": best_threshold,
        "label_source": sorted(df["label_source"].dropna().unique().tolist()),
        "hyperparameters": {**BASE_PARAMS, "scale_pos_weight": final_spw},
        "limitations": [
            "The target is a pseudo-label derived from the existing dashboard planning rule.",
            "The model currently learns to reproduce that rule, not real operator deployment success.",
            "Positive labels are geographically concentrated, especially in Kyauktan.",
            "The dataset is small (194 rows), so metrics are prototype evidence only.",
            "Operational use needs real operator KPI/RF, land, power/backhaul and permitting data."
        ],
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    pooled["versions"] = {
        "python": sys.version.split()[0],
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    pooled["reload_validation_max_abs_probability_difference"] = reload_diff
    (models_dir / "xgboost_metrics.json").write_text(
        json.dumps(pooled, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "model": str(model_path.relative_to(root)),
        "roc_auc_oof": pooled["roc_auc_oof"],
        "average_precision_oof": pooled["average_precision_oof"],
        "threshold": best_threshold,
        "f1_at_threshold": pooled["provisional_threshold"]["f1"],
        "reload_max_abs_diff": reload_diff,
    }, indent=2))


if __name__ == "__main__":
    main()
