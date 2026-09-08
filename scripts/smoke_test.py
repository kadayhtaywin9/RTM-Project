"""Offline smoke test for packaged models and local hazard fallbacks."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.hazard_engine import get_hazard_engine
from site_ai import recommend_sites_xgb


def main() -> None:
    candidates = pd.read_csv(ROOT / "data" / "candidate_village_tracts.csv")
    towers = pd.read_csv(ROOT / "data" / "tower_sites_population.csv").head(250)
    recommendations = recommend_sites_xgb(candidates, n_sites=5)
    required = {"tower_location", "priority_score", "expected_population_coverage", "recommendation_reason"}
    assert len(recommendations) == 5
    assert required.issubset(recommendations.columns)

    engine = get_hazard_engine()
    for hazard_type in ["flood", "cyclone", "earthquake", "compound"]:
        result, run = engine.run(towers, hazard_type=hazard_type, mode="local")
        assert len(result) == len(towers)
        assert result["hazard_ai_score"].between(0, 1).all()
        assert {"risk_level", "population_affected", "recommended_action", "main_factors"}.issubset(result.columns)
        assert run["mode"] == "local"
        print(f"{hazard_type}: {len(result)} towers, max={result['hazard_ai_score'].max():.3f}")

    print("GeoVision AI smoke test passed.")


if __name__ == "__main__":
    main()

