#!/usr/bin/env python3
"""Prepare weakly supervised XGBoost tower-site training data.

This Step-1 script converts data/candidate_village_tracts.csv into
 data/tower_training_data.csv.

The initial label is intentionally a *bootstrap/pseudo-label*: it uses the same
interpretable planning factors as the dashboard's current recommend_sites()
logic, then marks the top quartile as optimal_site=1. Replace these labels later
with operator/engineering ground truth when it becomes available.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

WEIGHTS = {
    "gap_score": 0.45,
    "population_score": 0.30,
    "is_rural": 0.15,
    "safety_score": 0.10,
    "elevation_score": 0.15,
}
POSITIVE_QUANTILE = 0.75

REQUIRED_COLUMNS = [
    "candidate_id",
    "adm3_name",
    "adm4_name",
    "adm4_pcode",
    "lat",
    "lon",
    "gap_score",
    "population_score",
    "is_rural",
    "safety_score",
    "elevation_score",
    "nearest_tower_km",
    "population_2020",
    "earthquake_score",
    "cyclone_score",
    "flood_score",
    "elevation_m",
]

OUTPUT_COLUMNS = [
    "candidate_id",
    "adm3_name",
    "adm4_name",
    "adm4_pcode",
    "lat",
    "lon",
    "spatial_group",
    "gap_score",
    "population_score",
    "is_rural",
    "safety_score",
    "elevation_score",
    "nearest_tower_km",
    "population_2020",
    "earthquake_score",
    "cyclone_score",
    "flood_score",
    "elevation_m",
    "label_suitability_score",
    "optimal_site",
    "label_source",
]


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile_linear(values: Iterable[float], q: float) -> float:
    """Equivalent to the common linear-interpolation quantile definition."""
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("Cannot compute a quantile from an empty sequence.")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1.")
    if len(vals) == 1:
        return vals[0]
    position = (len(vals) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(vals) - 1)
    fraction = position - lower
    return vals[lower] + (vals[upper] - vals[lower]) * fraction


def suitability_score(row: Dict[str, str]) -> float:
    weight_sum = sum(WEIGHTS.values())
    values = {
        "gap_score": clip01(float(row["gap_score"])),
        "population_score": clip01(float(row["population_score"])),
        "is_rural": clip01(float(row["is_rural"])),
        "safety_score": clip01(float(row["safety_score"])),
        "elevation_score": clip01(float(row["elevation_score"])),
    }
    return 100.0 * sum(WEIGHTS[k] * values[k] for k in WEIGHTS) / weight_sum


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No candidate rows found in {path}")
    return rows


def prepare(input_path: Path, output_path: Path, summary_path: Path) -> dict:
    rows = load_rows(input_path)

    scores = [suitability_score(row) for row in rows]
    threshold = percentile_linear(scores, POSITIVE_QUANTILE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    positive = 0
    townships = set()

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row, score in zip(rows, scores):
            label = int(score >= threshold)
            positive += label
            spatial_group = (row.get("adm3_name") or "UNKNOWN").strip() or "UNKNOWN"
            townships.add(spatial_group)

            out = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
            out["spatial_group"] = spatial_group
            out["label_suitability_score"] = f"{score:.6f}"
            out["optimal_site"] = str(label)
            out["label_source"] = "dashboard_rule_top_quartile_v1"
            writer.writerow(out)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows": len(rows),
        "positive_optimal_site": positive,
        "negative_optimal_site": len(rows) - positive,
        "positive_rate": positive / len(rows),
        "label_quantile": POSITIVE_QUANTILE,
        "label_threshold": threshold,
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": sum(scores) / len(scores),
        "spatial_groups": len(townships),
        "weights_raw": WEIGHTS,
        "weights_normalized": {k: v / sum(WEIGHTS.values()) for k, v in WEIGHTS.items()},
        "model_features_step2": list(WEIGHTS.keys()),
        "warning": (
            "optimal_site is a pseudo-label derived from the current dashboard planning rule. "
            "It is suitable for an MVP/prototype, not a substitute for operator deployment/KPI ground truth."
        ),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/candidate_village_tracts.csv"),
        help="Candidate CSV produced by the GeoAI preprocessing pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tower_training_data.csv"),
        help="Output training CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/tower_training_data_summary.json"),
        help="Output summary JSON.",
    )
    args = parser.parse_args()
    summary = prepare(args.input, args.output, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
