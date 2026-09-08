"""Shared feature engineering for tower vulnerability and network importance."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_METADATA = PROJECT_ROOT / "models" / "multi_hazard_model_metadata.json"


@lru_cache(maxsize=1)
def feature_normalization_contract() -> dict[str, float | str]:
    """Load the fixed training-set normalizers used by every serving batch."""
    raw = json.loads(NORMALIZATION_METADATA.read_text(encoding="utf-8"))
    contract = raw.get("feature_normalization", {})
    required = {"cell_count_max", "population_column", "population_max"}
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(f"Model normalization metadata is missing: {missing}")
    cell_count_max = float(contract["cell_count_max"])
    population_max = float(contract["population_max"])
    if not np.isfinite(cell_count_max) or cell_count_max < 1.0:
        raise ValueError("cell_count_max must be a finite value greater than or equal to 1")
    if not np.isfinite(population_max) or population_max <= 0.0:
        raise ValueError("population_max must be a finite positive value")
    return {
        "cell_count_max": cell_count_max,
        "population_column": str(contract["population_column"]),
        "population_max": population_max,
    }


def _numeric_optional(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    values = frame[name] if name in frame.columns else pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(values, errors="coerce").fillna(default).astype(float)


def prepare_tower_features(towers: pd.DataFrame) -> pd.DataFrame:
    out = towers.copy()
    if "tower_id" not in out or out["tower_id"].duplicated().any():
        raise ValueError("tower_id must exist and be unique")
    normalization = feature_normalization_contract()
    cell_count_max = float(normalization["cell_count_max"])
    population_column = str(normalization["population_column"])
    population_max = float(normalization["population_max"])
    radios = out.get("radios", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    technology_count = radios.str.count(",") + 1
    out["tower_vulnerability"] = np.clip(
        0.74 - 0.12 * (technology_count - 1) - 0.10 * radios.str.contains("LTE").astype(float) - 0.03 * radios.str.contains("UMTS").astype(float),
        0.25,
        0.80,
    )
    cell_count = _numeric_optional(out, "cell_count", 1.0).clip(lower=1.0, upper=cell_count_max)
    out["tower_redundancy"] = np.log1p(cell_count) / np.log1p(cell_count_max)
    population = _numeric_optional(out, population_column, 0.0).clip(lower=0.0, upper=population_max)
    population_norm = np.log1p(population) / np.log1p(population_max)
    out["network_importance"] = np.clip(0.65 * population_norm + 0.35 * out["tower_redundancy"], 0, 1)
    return out
