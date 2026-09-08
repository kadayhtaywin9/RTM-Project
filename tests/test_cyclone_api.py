from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.cyclone_api import (
    JTWCClient,
    add_cyclone_context,
    cyclone_category,
    discover_jtwc_fst_urls,
    parse_jtwc_coordinate,
    parse_jtwc_fst,
)


def _fst_line(
    forecast_hour: int,
    latitude: str,
    longitude: str,
    wind: int,
    pressure: int,
    *,
    cycle: str = "2026090400",
    storm_name: str = "ALPHA",
) -> str:
    fields = [""] * 28
    fields[0:11] = [
        "IO",
        "01",
        cycle,
        "00",
        "JTWC",
        str(forecast_hour),
        latitude,
        longitude,
        str(wind),
        str(pressure),
        "TS",
    ]
    fields[27] = storm_name
    return ", ".join(fields)


def test_category_boundaries_and_invalid_values() -> None:
    assert cyclone_category(float("nan")) == "Depression"
    assert cyclone_category(-5) == "Depression"
    assert cyclone_category(34) == "Tropical Storm"
    assert cyclone_category(64) == "Category 1"
    assert cyclone_category(83) == "Category 2"
    assert cyclone_category(96) == "Category 3"
    assert cyclone_category(113) == "Category 4"
    assert cyclone_category(137) == "Category 5"


def test_optional_context_columns_are_safe() -> None:
    result = add_cyclone_context(pd.DataFrame({"tower_id": [1]}))
    assert result.loc[0, "wind_speed"] == 0.0
    assert math.isinf(result.loc[0, "distance_to_cyclone"])
    assert result.loc[0, "cyclone_category"] == "Depression"


@pytest.mark.parametrize(
    ("token", "latitude", "expected"),
    [("167N", True, 16.7), ("083S", True, -8.3), ("956E", False, 95.6), ("1234W", False, -123.4)],
)
def test_coordinate_parsing(token: str, latitude: bool, expected: float) -> None:
    assert parse_jtwc_coordinate(token, latitude=latitude) == pytest.approx(expected)


def test_parse_forecast_keeps_latest_cycle_and_deduplicates_radii() -> None:
    text = "\n".join(
        [
            _fst_line(0, "165N", "934E", 45, 990, cycle="2026090318"),
            _fst_line(0, "167N", "936E", 50, 986),
            _fst_line(0, "167N", "936E", 50, 986),
            _fst_line(24, "174N", "948E", 65, 972),
            "not, an, atcf, record",
        ]
    )

    result = parse_jtwc_fst(text, source_url="https://example.test/io012026.fst")

    assert result["forecast_hour"].tolist() == [0, 24]
    assert result["lat"].tolist() == pytest.approx([16.7, 17.4])
    assert result["lon"].tolist() == pytest.approx([93.6, 94.8])
    assert result["storm_name"].tolist() == ["ALPHA", "ALPHA"]
    assert result["is_forecast"].tolist() == [False, True]


def test_discovery_is_bounded_by_basin_and_year() -> None:
    html = "io012026.fst io022026.fst io032025.fst wp052026.fst sh012026.fst al012026.fst"
    urls = discover_jtwc_fst_urls(
        html,
        "https://example.test/JTWC/",
        years=(2026,),
        basins=("IO", "WP"),
        max_files_per_basin=1,
    )
    assert urls == [
        "https://example.test/JTWC/io022026.fst",
        "https://example.test/JTWC/wp052026.fst",
    ]


def test_nrl_warning_links_discover_matching_forecast_products() -> None:
    html = """
        <a href="docs/current_storms/io012026.wrn">IO warning</a>
        <a href="docs/current_storms/wp222026.wrn">WP warning</a>
        <a href="docs/current_storms/sh012026.wrn">SH warning</a>
    """

    urls = discover_jtwc_fst_urls(
        html,
        "https://science.nrlmry.navy.mil/atcf/docs/current_storms/",
        years=(2026,),
        basins=("IO", "WP"),
    )

    assert urls == [
        "https://science.nrlmry.navy.mil/atcf/docs/current_storms/io012026.fst",
        "https://science.nrlmry.navy.mil/atcf/docs/current_storms/wp222026.fst",
    ]


def test_client_rejects_http_200_waf_page_instead_of_reporting_no_storm() -> None:
    client = JTWCClient(
        index_url="https://example.test/atcf/index1.html",
        fetch_text=lambda url: "<html><h1>Request Rejected</h1><p>The requested URL was rejected.</p></html>",
    )

    with pytest.raises(RuntimeError, match="rejected or empty ATCF index"):
        client.active_track(now="2026-09-04T06:00:00Z")


def test_valid_atcf_index_without_configured_basin_is_no_active_storm() -> None:
    client = JTWCClient(
        index_url="https://example.test/atcf/index1.html",
        basins=("IO", "WP"),
        fetch_text=lambda url: (
            "<html><title>Automated Tropical Cyclone Forecasting System</title>"
            '<h2>Current Global Tropical Cyclone Activity</h2><a href="sh012026.wrn">SH</a></html>'
        ),
    )

    result = client.active_track(now="2026-09-04T06:00:00Z")

    assert result.empty


def test_generic_atcf_outage_page_is_not_reported_as_no_active_storm() -> None:
    client = JTWCClient(
        index_url="https://example.test/atcf/index1.html",
        fetch_text=lambda url: "<html><title>ATCF service unavailable</title></html>",
    )

    with pytest.raises(RuntimeError, match="unrecognized ATCF index"):
        client.active_track(now="2026-09-04T06:00:00Z")


def test_client_uses_separate_nrl_index_and_product_urls() -> None:
    index_url = "https://example.test/atcf/index1.html"
    product_url = "https://example.test/atcf/docs/current_storms/io012026.fst"
    requested: list[str] = []

    def fetch(url: str) -> str:
        requested.append(url)
        if url == index_url:
            return '<html><title>ATCF current storms</title><a href="docs/current_storms/io012026.wrn">IO</a></html>'
        if url == product_url:
            return _fst_line(0, "165N", "960E", 55, 982)
        raise AssertionError(f"Unexpected URL: {url}")

    client = JTWCClient(
        base_url="https://example.test/atcf/docs/current_storms/",
        index_url=index_url,
        fetch_text=fetch,
    )

    track = client.active_track(now="2026-09-04T06:00:00Z")

    assert requested == [index_url, product_url]
    assert track["storm_id"].tolist() == ["IO012026"]


def test_explicit_custom_base_remains_the_discovery_index_by_default() -> None:
    client = JTWCClient(base_url="https://example.test/custom-mirror/")

    assert client.base_url == "https://example.test/custom-mirror/"
    assert client.index_url == client.base_url


def test_incomplete_advertised_products_fail_closed() -> None:
    index_url = "https://example.test/atcf/index1.html"
    base_url = "https://example.test/atcf/docs/current_storms/"

    def fetch(url: str) -> str:
        if url == index_url:
            return '<html><a href="io012026.wrn">IO 01</a><a href="io022026.wrn">IO 02</a></html>'
        if url.endswith("io022026.fst"):
            raise RuntimeError("temporary upstream failure")
        return _fst_line(0, "165N", "960E", 55, 982, cycle="2026080100")

    client = JTWCClient(base_url=base_url, index_url=index_url, fetch_text=fetch)

    with pytest.raises(RuntimeError, match="incomplete current-storm data"):
        client.active_track(now="2026-09-04T06:00:00Z")


def test_client_builds_current_and_forecast_tower_signal_without_key() -> None:
    product = "\n".join(
        [
            _fst_line(0, "165N", "960E", 55, 982),
            _fst_line(24, "170N", "962E", 70, 968),
        ]
    )
    requested: list[str] = []

    def fetch(url: str) -> str:
        requested.append(url)
        return product

    towers = pd.DataFrame({"tower_id": [1, 2], "lat": [16.8, 17.4], "lon": [96.1, 95.8]})
    client = JTWCClient(
        base_url="https://example.test/JTWC/",
        storm_ids=("IO012026",),
        fetch_text=fetch,
    )

    signal, metadata, context = client.tower_features(towers, now="2026-09-04T06:00:00Z")

    assert requested == ["https://example.test/JTWC/io012026.fst"]
    assert metadata["data_status"] == "active"
    assert metadata["active_storm_count"] == 1
    assert len(metadata["forecast_track"]) == 2
    assert np.all((signal >= 0) & (signal <= 1))
    assert signal.max() > 0
    assert context["track_status"].eq("Active JTWC forecast").all()


def test_stale_product_is_unknown_not_no_active_storm() -> None:
    client = JTWCClient(
        base_url="https://example.test/JTWC/",
        storm_ids=("IO012026",),
        fetch_text=lambda url: _fst_line(0, "165N", "960E", 55, 982, cycle="2026080100"),
    )
    towers = pd.DataFrame({"tower_id": [1], "lat": [16.8], "lon": [96.1]})

    with pytest.raises(RuntimeError, match="current storm status is unknown"):
        client.tower_features(towers, now="2026-09-04T06:00:00Z")


@pytest.mark.parametrize("link", ["", '<a href="io012026.wrn">IO</a>'])
def test_stale_index_cannot_verify_current_activity(link: str) -> None:
    client = JTWCClient(fetch_text=lambda url: (
        "<title>Automated Tropical Cyclone Forecasting System</title>"
        "<h2>Current Global Tropical Cyclone Activity</h2>"
        f"Last updated Tue Sep 01 06:00:00 UTC 2026 {link}"
    ))
    with pytest.raises(RuntimeError, match="index is too stale"):
        client.active_track(now="2026-09-04T06:00:00Z")


def test_fresh_empty_official_index_has_timestamp_metadata() -> None:
    client = JTWCClient(fetch_text=lambda url: (
        "<title>Automated Tropical Cyclone Forecasting System</title>"
        "<h2>Current Global Tropical Cyclone Activity</h2>"
        "Last updated Fri Sep 04 00:00:00 UTC 2026"
    ))
    result = client.active_track(now="2026-09-04T06:00:00Z")
    assert result.empty
    assert result.attrs["source_metadata"]["index_age_hours"] == 6.0


def test_southern_hemisphere_next_season_is_discovered_and_parsed() -> None:
    product = _fst_line(0, "083S", "956E", 45, 990).replace("IO, 01", "SH, 01")
    client = JTWCClient(basins=("SH",), fetch_text=lambda url: (
        '<a href="sh012027.wrn">SH</a>' if url.endswith("index1.html") else product
    ))
    result = client.active_track(now="2026-09-04T06:00:00Z")
    assert result["storm_id"].tolist() == ["SH012027"]
