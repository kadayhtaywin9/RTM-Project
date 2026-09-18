"""Offline input-sensitivity audit, not observed-accuracy or fairness validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import hazard_ai


def main():
    towers = pd.read_csv(BASE / "data/tower_sites_population.csv")
    results = {}
    for kind in ("earthquake", "cyclone"):
        predict = getattr(hazard_ai, f"_predict_{kind}")
        zero = predict(towers, np.zeros(len(towers)), "Offline sensitivity test")
        high = predict(towers, np.ones(len(towers)), "Offline sensitivity test")
        results[kind] = {
            "tower_count": len(zero),
            "zero_event_median_score_0_to_100": float(zero.hazard_ai_score.median() * 100),
            "zero_event_at_or_above_50_count": int(zero.hazard_ai_score.ge(.5).sum()),
            "maximum_event_input_median_score_0_to_100": float(high.hazard_ai_score.median() * 100),
            "endpoint_monotonicity_violations": int((high.hazard_ai_score < zero.hazard_ai_score - 1e-6).sum()),
        }
    report = {
        "scope": "Packaged models; changing only event_intensity from zero to one. No external APIs.",
        "interpretation": "High zero-event scores reflect historical/site proxy assumptions, not current disasters. Endpoint checks do not establish full monotonicity, accuracy or absence of bias.",
        "models": results,
        "observed_outage_validation": "Not available",
        "bias_validation": "Not established; independent labels and representative geographic/event holdouts required.",
    }
    (BASE / "HAZARD_MODEL_AUDIT_V20.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
