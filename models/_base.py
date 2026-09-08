"""Reusable, validated XGBoost loading, prediction, and TreeSHAP inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from utils.caching import cache_resource

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@cache_resource(max_entries=8)
def _load_model(path_text: str, task: str) -> XGBClassifier | XGBRegressor:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"GeoVision model artifact not found: {path}")
    model: XGBClassifier | XGBRegressor
    model = XGBClassifier() if task == "classification" else XGBRegressor()
    model.load_model(path)
    return model


class TreeModel:
    """Small serving wrapper with a strict feature contract."""

    task = "regression"

    def __init__(self, model_file: str, metadata_file: str, metadata_path: tuple[str, ...] = ()) -> None:
        self.model_path = PROJECT_ROOT / "models" / model_file
        raw = json.loads((PROJECT_ROOT / "models" / metadata_file).read_text(encoding="utf-8"))
        for key in metadata_path:
            raw = raw[key]
        self.metadata: dict[str, Any] = raw
        self.features = list(raw.get("features_in_order", []))
        if not self.features:
            raise ValueError(f"No feature contract found for {model_file}")

    @property
    def model(self) -> XGBClassifier | XGBRegressor:
        return _load_model(str(self.model_path), self.task)

    def feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        clean = frame[self.features].apply(pd.to_numeric, errors="coerce")
        if clean.isna().any().any():
            missing_counts = clean.isna().sum()
            bad = {k: int(v) for k, v in missing_counts.items() if v}
            raise ValueError(f"Null or non-numeric model features: {bad}")
        return clean.astype(float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.feature_frame(frame)
        if self.task == "classification":
            values = self.model.predict_proba(features)[:, 1]  # type: ignore[union-attr]
        else:
            values = self.model.predict(features)
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)

    def tree_shap(self, frame: pd.DataFrame) -> np.ndarray:
        """Return exact native TreeSHAP margin contributions plus the bias column."""
        features = self.feature_frame(frame)
        matrix = xgb.DMatrix(features, feature_names=self.features)
        return np.asarray(self.model.get_booster().predict(matrix, pred_contribs=True), dtype=float)

    def explanation_strings(self, frame: pd.DataFrame, top_n: int = 3) -> list[str]:
        contrib = self.tree_shap(frame)[:, :-1]
        output: list[str] = []
        for row in contrib:
            order = np.argsort(np.abs(row))[::-1][:top_n]
            parts = [f"{'+' if row[i] >= 0 else '-'} {self.features[i].replace('_', ' ')}" for i in order]
            output.append("; ".join(parts))
        return output

