"""Tower Recommendation AI serving contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import TreeModel


class TowerRecommendationModel(TreeModel):
    task = "classification"

    def __init__(self) -> None:
        super().__init__("tower_site_xgb.json", "model_metadata.json")

    def rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        out = candidates.copy()
        out["ai_probability"] = self.predict(out)
        out["priority_score"] = 100.0 * out["ai_probability"]
        out["suitability_score"] = out["priority_score"]
        out["tower_location"] = out.apply(lambda r: f"{float(r['lat']):.6f}, {float(r['lon']):.6f}", axis=1)
        # Admin-4 population is a planning proxy, not a radio-propagation estimate.
        out["expected_population_coverage"] = pd.to_numeric(
            out.get("population_2020", pd.Series(0.0, index=out.index)), errors="coerce"
        ).fillna(0.0)
        out["recommendation_reason"] = self.explanation_strings(out)
        threshold = float(self.metadata.get("provisional_decision_threshold", 0.65))
        out["ai_decision"] = np.where(out["ai_probability"] >= threshold, "OPTIMAL CANDIDATE", "NOT OPTIMAL")
        out["recommendation_engine"] = "GeoVision AI"
        return out

