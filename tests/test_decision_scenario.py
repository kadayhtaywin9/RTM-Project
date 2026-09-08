from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.decision_engine import MODEL_TYPES, DecisionEngine, attach_coverage_scenario


class _ExplanationModel:
    def explanation_strings(self, frame: pd.DataFrame, top_n: int = 3) -> list[str]:
        return ["+ historical exposure"] * len(frame)


def _result(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setitem(MODEL_TYPES, "cyclone", _ExplanationModel)
    return DecisionEngine().enrich(pd.DataFrame({
        "tower_id": [0, 1],
        "hazard_ai_score": [0.8, 0.2],
        "hazard_class": ["Very High", "Low"],
        "estimated_population_primary_5km": [1000, 400],
        "tower_redundancy": [0.5, 0.8],
    }), "cyclone")


def _loads() -> pd.DataFrame:
    return pd.DataFrame({
        "tower_id": [1, 0],
        "baseline_people_within_radius": [200.0, 600.0],
        "population_directly_affected": [0.0, 600.0],
        "population_rerouted": [0.0, 500.0],
        "population_losing_coverage": [0.0, 100.0],
    })


def test_score_does_not_invent_expected_people_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _result(monkeypatch)
    assert result["population_affected"].isna().all()
    assert result["coverage_impact"].isna().all()
    assert result["impact_status"].eq("requires_coverage_scenario").all()


def test_people_counts_follow_actual_scenario_by_tower_id(monkeypatch: pytest.MonkeyPatch) -> None:
    result = attach_coverage_scenario(_result(monkeypatch), _loads())
    assert result["tower_id"].tolist() == [0, 1]
    assert result["baseline_population_served"].tolist() == [600, 200]
    assert result["population_affected"].tolist() == [600, 0]
    assert result["population_rerouted"].tolist() == [500, 0]
    assert result["coverage_impact"].tolist() == [100, 0]
    assert result["impact_status"].eq("coverage_scenario").all()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "negative", "nan", "conservation"])
def test_incomplete_or_inconsistent_counts_are_rejected(monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    loads = _loads()
    if invalid == "missing":
        loads = loads.iloc[:1]
    elif invalid == "duplicate":
        loads.loc[0, "tower_id"] = 0
    elif invalid == "negative":
        loads.loc[0, "population_directly_affected"] = -1
    elif invalid == "nan":
        loads.loc[0, "population_directly_affected"] = np.nan
    else:
        loads.loc[1, "population_losing_coverage"] = 200
    with pytest.raises(ValueError):
        attach_coverage_scenario(_result(monkeypatch), loads)
