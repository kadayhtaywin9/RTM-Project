from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from geoai_engine import simulate_population_coverage


def _network() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Eleven towers; the only survivor is outside the ten-neighbor cache."""
    tower_x = 200_000.0 + np.arange(11) * 100.0
    pixel_x = 200_000.0 + np.array([-50.0, 1020.0, 10_000.0])
    to_wgs84 = Transformer.from_crs(32647, 4326, always_xy=True)
    tower_lon, tower_lat = to_wgs84.transform(tower_x, np.full(11, 1_860_000.0))
    pixel_lon, pixel_lat = to_wgs84.transform(pixel_x, np.full(3, 1_860_000.0))
    distances = np.abs(pixel_x[:, None] - tower_x[None, :]) / 1000.0
    nearest = np.argsort(distances, axis=1)[:, :10]
    grid = {
        "population": np.array([100.0, 200.0, 50.0]),
        "area_id": np.array([1, 1, 1]),
        "lon": pixel_lon,
        "lat": pixel_lat,
        "nearest_tower_idx": nearest,
        "nearest_tower_dist_km": np.take_along_axis(distances, nearest, axis=1),
    }
    towers = pd.DataFrame({
        "tower_id": np.arange(11),
        "scenario_risk": [1.0] * 10 + [0.0],
        "analysis_area": ["A"] * 11,
        "lon": tower_lon,
        "lat": tower_lat,
    })
    return grid, towers


def _run(grid, towers, radius=2.0):
    return simulate_population_coverage(grid, towers, [1], ["A"], radius, 0.5)


def test_reroutes_to_eleventh_tower_beyond_cached_neighbors() -> None:
    grid, towers = _network()
    assert 10 not in grid["nearest_tower_idx"][0]
    metrics, load = _run(grid, towers)

    assert metrics["baseline_served"] == 300.0
    assert metrics["post_served"] == 300.0
    assert metrics["population_rerouted"] == 100.0
    assert metrics["population_losing_coverage"] == 0.0
    # Attribute impact to the ORIGINAL tower, and post-event load to its survivor.
    assert load.loc[0, "population_directly_affected"] == 100.0
    assert load.loc[0, "population_rerouted"] == 100.0
    assert load.loc[10, "population_rerouted"] == 0.0
    assert load.loc[10, "post_disaster_people_within_radius"] == 300.0


def test_all_failed_counts_only_previously_covered_population() -> None:
    grid, towers = _network()
    towers["scenario_risk"] = 1.0
    metrics, load = _run(grid, towers)

    assert metrics["population_total"] == 350.0
    assert metrics["baseline_uncovered"] == 50.0
    assert metrics["post_served"] == 0.0
    assert metrics["population_directly_affected"] == 300.0
    assert metrics["population_losing_coverage"] == 300.0
    assert metrics["population_rerouted"] == 0.0
    assert metrics["failed_towers"] == 11
    assert load["post_disaster_people_within_radius"].sum() == 0.0


def test_no_failures_preserves_population_assignments() -> None:
    grid, towers = _network()
    towers["scenario_risk"] = 0.0
    metrics, load = _run(grid, towers)

    assert metrics["baseline_served"] == metrics["post_served"] == 300.0
    assert metrics["population_directly_affected"] == 0.0
    assert metrics["population_rerouted"] == 0.0
    assert metrics["population_losing_coverage"] == 0.0
    np.testing.assert_array_equal(
        load["baseline_people_within_radius"], load["post_disaster_people_within_radius"],
    )


@pytest.mark.parametrize("radius, rerouted, lost", [(0.5, 0.0, 100.0), (2.0, 100.0, 0.0)])
def test_radius_controls_coverage_and_conserves_all_impact_totals(radius, rerouted, lost) -> None:
    grid, towers = _network()
    metrics, load = _run(grid, towers, radius)

    assert metrics["population_rerouted"] == rerouted
    assert metrics["population_losing_coverage"] == lost
    assert metrics["population_directly_affected"] == rerouted + lost
    assert metrics["population_losing_coverage"] == metrics["baseline_served"] - metrics["post_served"]
    assert metrics["population_total"] == metrics["post_served"] + metrics["post_uncovered"]
    assert load["baseline_people_within_radius"].sum() == metrics["baseline_served"]
    assert load["post_disaster_people_within_radius"].sum() == metrics["post_served"]
    for column in ("population_directly_affected", "population_rerouted", "population_losing_coverage"):
        assert load[column].sum() == metrics[column]
    np.testing.assert_array_equal(
        load["population_directly_affected"],
        load["population_rerouted"] + load["population_losing_coverage"],
    )


def test_reordered_tower_rows_preserve_ids_and_results() -> None:
    grid, towers = _network()
    expected_metrics, expected_load = _run(grid, towers)
    reordered = towers.sample(frac=1.0, random_state=42).reset_index(drop=True)
    metrics, load = _run(grid, reordered)

    assert load["tower_id"].tolist() == reordered["tower_id"].tolist()
    assert metrics == expected_metrics
    pd.testing.assert_frame_equal(
        load.sort_values("tower_id").reset_index(drop=True), expected_load,
    )


def test_unselected_area_towers_remain_available_for_selected_population() -> None:
    grid, towers = _network()
    towers.loc[10, ["analysis_area", "scenario_risk"]] = ["B", 1.0]
    grid["area_id"][1:] = 2
    metrics, load = _run(grid, towers)

    assert metrics["population_total"] == 100.0
    assert metrics["population_rerouted"] == 100.0
    assert metrics["population_losing_coverage"] == 0.0
    assert not load.loc[10, "failed_in_scenario"]
    assert load.loc[10, "post_disaster_people_within_radius"] == 100.0


@pytest.mark.parametrize("invalid_ids", [list(range(10)) + [9], list(range(10)) + [11], [0.5] + list(range(1, 11))])
def test_rejects_duplicate_missing_or_noninteger_tower_ids(invalid_ids) -> None:
    grid, towers = _network()
    towers["tower_id"] = invalid_ids
    with pytest.raises(ValueError, match="complete tower table"):
        _run(grid, towers)


def test_rejects_cache_id_outside_full_tower_table() -> None:
    grid, towers = _network()
    grid["nearest_tower_idx"][0, 0] = -1
    with pytest.raises(ValueError, match="outside the complete tower table"):
        _run(grid, towers)


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf])
def test_rejects_invalid_radius(radius) -> None:
    grid, towers = _network()
    with pytest.raises(ValueError, match="service_radius_km"):
        _run(grid, towers, radius)


def test_rejects_missing_population_and_invalid_risk_instead_of_silent_totals() -> None:
    grid, towers = _network()
    grid["population"][0] = np.nan
    with pytest.raises(ValueError, match="Population values"):
        _run(grid, towers)
    grid["population"][0] = 100.0
    towers.loc[0, "scenario_risk"] = np.nan
    with pytest.raises(ValueError, match="scenario_risk"):
        _run(grid, towers)


def test_no_selected_population_returns_empty_result() -> None:
    grid, towers = _network()
    metrics, load = simulate_population_coverage(grid, towers, [99], ["A"], 2.0, 0.5)
    assert metrics == {}
    assert load.empty
