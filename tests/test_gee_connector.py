from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data import gee_connector


def test_dynamic_world_contract_constants() -> None:
    assert gee_connector.DYNAMIC_WORLD_COLLECTION_ID == "GOOGLE/DYNAMICWORLD/V1"
    assert gee_connector.DYNAMIC_WORLD_LOOKBACK_DAYS == 180
    assert gee_connector.DYNAMIC_WORLD_CACHE_TTL_SECONDS > 0
    assert gee_connector.DYNAMIC_WORLD_MIN_CONFIDENCE == 0.55
    assert gee_connector.DYNAMIC_WORLD_SAMPLE_SCALE_METERS == 10
    assert gee_connector.DYNAMIC_WORLD_PROBABILITY_BANDS == (
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    )
    assert gee_connector.STATIC_FEATURE_COLUMNS == (
        "tower_id",
        "elevation",
        "slope",
        "land_cover",
        "land_cover_confidence",
    )


def test_static_features_preserve_tower_order_and_masked_samples(monkeypatch: Any) -> None:
    towers = pd.DataFrame(
        {
            "tower_id": [20, 10, 30],
            "lat": [16.8, 16.9, 17.0],
            "lon": [96.1, 96.2, 96.3],
        }
    )
    terrain = pd.DataFrame(
        {
            "tower_id": [10, 20],
            "elevation": [8.0, 14.0],
            "slope": [0.5, 1.5],
        }
    )
    land_cover = pd.DataFrame(
        {
            "tower_id": [20, 30],
            "land_cover": [6, 4],
            "land_cover_confidence": [0.93, 0.81],
        }
    )
    monkeypatch.setattr(gee_connector, "_terrain_features", lambda records, project, credentials: terrain)
    monkeypatch.setattr(
        gee_connector,
        "_dynamic_world_features",
        lambda records, project, credentials: land_cover,
    )

    result = gee_connector.EarthEngineConnector(project_id="test-project").static_features(towers)

    assert list(result.columns) == list(gee_connector.STATIC_FEATURE_COLUMNS)
    assert result["tower_id"].tolist() == [20, 10, 30]
    assert result.loc[result["tower_id"] == 20, "land_cover_confidence"].item() == 0.93
    assert pd.isna(result.loc[result["tower_id"] == 10, "land_cover"].item())
    assert pd.isna(result.loc[result["tower_id"] == 30, "elevation"].item())


@dataclass
class _FakeImage:
    selections: list[Any] = field(default_factory=list)
    reducers: list[Any] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    added_bands: int = 0

    def select(self, bands: Any) -> _FakeImage:
        self.selections.append(bands)
        return self

    def reduce(self, reducer: Any) -> _FakeImage:
        self.reducers.append(reducer)
        return self

    def rename(self, name: str) -> _FakeImage:
        self.names.append(name)
        return self

    def addBands(self, other: Any) -> _FakeImage:
        assert isinstance(other, _FakeImage)
        self.added_bands += 1
        return self


@dataclass
class _FakeCollection:
    image: _FakeImage = field(default_factory=_FakeImage)
    date_window: tuple[str, str] | None = None
    bounds: Any = None
    aggregate_value: int | None = 1_725_408_000_000
    quality_band: str | None = None
    final_selection: Any = None
    sample_kwargs: dict[str, Any] | None = None

    def filterDate(self, start: str, end: str) -> _FakeCollection:
        self.date_window = (start, end)
        return self

    def filterBounds(self, bounds: Any) -> _FakeCollection:
        self.bounds = bounds
        return self

    def aggregate_max(self, name: str) -> _FakeScalar:
        assert name == "system:time_start"
        return _FakeScalar(self.aggregate_value)

    def map(self, callback: Any) -> _FakeCollection:
        callback(self.image)
        return self

    def qualityMosaic(self, band: str) -> _FakeCollection:
        self.quality_band = band
        return self

    def select(self, bands: Any) -> _FakeCollection:
        self.final_selection = bands
        return self

    def sampleRegions(self, **kwargs: Any) -> _FakeCollection:
        self.sample_kwargs = kwargs
        return self


@dataclass
class _FakeScalar:
    value: Any

    def getInfo(self) -> Any:
        return self.value


class _FakeReducer:
    @staticmethod
    def max() -> str:
        return "max-reducer"


class _FakeEE:
    Reducer = _FakeReducer

    def __init__(self) -> None:
        self.collection_id: str | None = None
        self.collection = _FakeCollection()

    def ImageCollection(self, collection_id: str) -> _FakeCollection:
        self.collection_id = collection_id
        return self.collection


class _FakePoints:
    def geometry(self) -> str:
        return "tower-region"


def test_dynamic_world_query_uses_recent_confidence_quality_mosaic(monkeypatch: Any) -> None:
    fake_ee = _FakeEE()
    points = _FakePoints()
    monkeypatch.setattr(gee_connector, "initialize_earth_engine", lambda project, credentials: fake_ee)
    monkeypatch.setattr(gee_connector, "_point_collection", lambda ee, records: points)
    monkeypatch.setattr(
        gee_connector,
        "_features_to_frame",
        lambda ee, sampled: pd.DataFrame(
            {"tower_id": [7], "land_cover": [6], "land_cover_confidence": [0.88]}
        ),
    )

    result = gee_connector._dynamic_world_features.__wrapped__(
        ((7, 16.8, 96.1),),
        "test-project",
        None,
    )

    collection = fake_ee.collection
    assert fake_ee.collection_id == gee_connector.DYNAMIC_WORLD_COLLECTION_ID
    assert collection.bounds == "tower-region"
    assert collection.date_window is not None
    start, end = map(pd.Timestamp, collection.date_window)
    assert (end - start).days == gee_connector.DYNAMIC_WORLD_LOOKBACK_DAYS + 1
    assert list(gee_connector.DYNAMIC_WORLD_PROBABILITY_BANDS) in collection.image.selections
    assert "label" in collection.image.selections
    assert "max-reducer" in collection.image.reducers
    assert collection.quality_band == "land_cover_confidence"
    assert collection.final_selection == ["land_cover", "land_cover_confidence"]
    assert collection.sample_kwargs == {
        "collection": points,
        "properties": ["tower_id"],
        "scale": gee_connector.DYNAMIC_WORLD_SAMPLE_SCALE_METERS,
        "geometries": False,
        "tileScale": 4,
    }
    assert result.to_dict("records") == [
        {"tower_id": 7, "land_cover": 6, "land_cover_confidence": 0.88}
    ]


def test_empty_static_features_keep_public_schema(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        gee_connector,
        "_terrain_features",
        lambda records, project, credentials: gee_connector._feature_schema(
            pd.DataFrame(), gee_connector.TERRAIN_FEATURE_COLUMNS
        ),
    )
    monkeypatch.setattr(
        gee_connector,
        "_dynamic_world_features",
        lambda records, project, credentials: gee_connector._feature_schema(
            pd.DataFrame(), gee_connector.DYNAMIC_WORLD_FEATURE_COLUMNS
        ),
    )
    towers = pd.DataFrame(columns=["tower_id", "lat", "lon"])

    result = gee_connector.EarthEngineConnector().static_features(towers)

    assert result.empty
    assert list(result.columns) == list(gee_connector.STATIC_FEATURE_COLUMNS)


def test_dynamic_world_hides_low_confidence_label(monkeypatch: Any) -> None:
    fake_ee = _FakeEE()
    monkeypatch.setattr(gee_connector, "initialize_earth_engine", lambda project, credentials: fake_ee)
    monkeypatch.setattr(gee_connector, "_point_collection", lambda ee, records: _FakePoints())
    monkeypatch.setattr(
        gee_connector,
        "_features_to_frame",
        lambda ee, sampled: pd.DataFrame(
            {
                "tower_id": [7, 8],
                "land_cover": [6, 4],
                "land_cover_confidence": [0.91, 0.40],
            }
        ),
    )

    result = gee_connector._dynamic_world_features.__wrapped__(
        ((7, 16.8, 96.1), (8, 16.9, 96.2)),
        "test-project",
        None,
    )

    assert result.loc[result["tower_id"] == 7, "land_cover"].item() == 6
    assert pd.isna(result.loc[result["tower_id"] == 8, "land_cover"].item())


def _valid_rainfall_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tower_id": [2, 1],
            "rainfall_24h": [2.0, 1.0],
            "rainfall_72h": [4.0, 3.0],
            "rainfall_30d": [20.0, 10.0],
            "elevation": [7.0, 8.0],
        }
    )


def test_rainfall_validator_preserves_context_and_requested_order() -> None:
    result = gee_connector.validate_rainfall_samples(
        _valid_rainfall_frame(),
        [1, 2],
        "2026-09-06T00:00:00Z",
        now="2026-09-06T12:00:00Z",
    )

    assert result["tower_id"].tolist() == [1, 2]
    assert result["elevation"].tolist() == [8.0, 7.0]
    assert list(result.columns) == [
        "tower_id",
        "rainfall_24h",
        "rainfall_72h",
        "rainfall_30d",
        "elevation",
    ]


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        ("2026-09-03T23:59:59Z", "stale"),
        ("2026-09-06T13:01:00Z", "future"),
    ],
)
def test_rainfall_validator_rejects_untrustworthy_source_time(timestamp: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        gee_connector.validate_rainfall_samples(
            _valid_rainfall_frame(),
            [1, 2],
            timestamp,
            now="2026-09-06T12:00:00Z",
        )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "missing required columns"),
        (_valid_rainfall_frame().drop(columns="rainfall_72h"), "missing required columns"),
        (_valid_rainfall_frame().iloc[[0]], "incomplete tower coverage"),
        (pd.concat([_valid_rainfall_frame(), _valid_rainfall_frame().iloc[[0]]]), "duplicate tower IDs"),
        (_valid_rainfall_frame().assign(rainfall_24h=[np.nan, 1.0]), "non-finite"),
        (_valid_rainfall_frame().assign(rainfall_72h=[-1.0, 3.0]), "negative"),
    ],
)
def test_rainfall_validator_rejects_incomplete_or_invalid_samples(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        gee_connector.validate_rainfall_samples(
            frame,
            [1, 2],
            "2026-09-06T00:00:00Z",
            now="2026-09-06T12:00:00Z",
        )


def test_rainfall_validator_rejects_duplicate_columns_and_unrequested_towers() -> None:
    duplicate_columns = _valid_rainfall_frame()
    duplicate_columns.columns = [
        "tower_id",
        "rainfall_24h",
        "rainfall_72h",
        "rainfall_30d",
        "rainfall_30d",
    ]
    with pytest.raises(RuntimeError, match="duplicate columns"):
        gee_connector.validate_rainfall_samples(
            duplicate_columns,
            [1, 2],
            "2026-09-06T00:00:00Z",
            now="2026-09-06T12:00:00Z",
        )

    unrequested = _valid_rainfall_frame().assign(tower_id=[1, 3])
    with pytest.raises(RuntimeError, match="unrequested tower IDs"):
        gee_connector.validate_rainfall_samples(
            unrequested,
            [1, 2],
            "2026-09-06T00:00:00Z",
            now="2026-09-06T12:00:00Z",
        )


def test_hourly_coverage_requires_every_unique_hour_for_thirty_days() -> None:
    latest = pd.Timestamp("2026-09-06T00:00:00Z")
    complete = (pd.date_range(end=latest, periods=720, freq="h").asi8 // 1_000_000).tolist()

    assert gee_connector.validate_gsmap_hourly_coverage(complete, complete[-1]) == {
        24: 24,
        72: 72,
        720: 720,
    }

    with pytest.raises(RuntimeError, match="incomplete"):
        gee_connector.validate_gsmap_hourly_coverage(complete[:-1], complete[-1])
    with pytest.raises(RuntimeError, match="duplicate"):
        gee_connector.validate_gsmap_hourly_coverage([*complete, complete[-1]], complete[-1])


def test_public_rainfall_path_revalidates_cached_source_age(monkeypatch: Any) -> None:
    towers = pd.DataFrame({"tower_id": [1, 2], "lat": [16.8, 16.9], "lon": [96.1, 96.2]})
    monkeypatch.setattr(
        gee_connector,
        "_rainfall_features",
        lambda records, project, credentials: (_valid_rainfall_frame(), "2000-01-01T00:00:00Z"),
    )

    with pytest.raises(RuntimeError, match="stale"):
        gee_connector.EarthEngineConnector().rainfall_features(towers)


@dataclass
class _FakeRainDate:
    value: Any

    def advance(self, amount: int, unit: str) -> _FakeRainDate:
        return self


@dataclass
class _FakeRainBand:
    name: str = ""
    mask: Any = None

    def updateMask(self, mask: Any) -> _FakeRainBand:
        self.mask = mask
        return self

    def rename(self, name: str) -> _FakeRainBand:
        self.name = name
        return self


class _FakeRainCollection:
    def __init__(self, latest_ms: int, inventory: list[int]) -> None:
        self.latest_ms = latest_ms
        self.inventory = inventory
        self.selected_band: str | None = None

    def select(self, band: str) -> _FakeRainCollection:
        self.selected_band = band
        return self

    def filterDate(self, start: Any, end: Any) -> _FakeRainCollection:
        return self

    def aggregate_max(self, name: str) -> _FakeScalar:
        return _FakeScalar(self.latest_ms)

    def aggregate_array(self, name: str) -> _FakeScalar:
        return _FakeScalar(self.inventory)

    def sum(self) -> _FakeRainBand:
        return _FakeRainBand()

    def count(self) -> _FakePixelCount:
        return _FakePixelCount()


class _FakePixelCount:
    def eq(self, hours: int) -> int:
        return hours


class _FakeRainStack:
    def sampleRegions(self, **kwargs: Any) -> _FakeRainStack:
        return self


class _FakeRainImageFactory:
    @staticmethod
    def cat(images: list[_FakeRainBand]) -> _FakeRainStack:
        assert [item.name for item in images] == ["rainfall_24h", "rainfall_72h", "rainfall_30d"]
        assert [item.mask for item in images] == [24, 72, 720]
        return _FakeRainStack()


class _FakeRainEE:
    Image = _FakeRainImageFactory

    def __init__(self, latest_ms: int, inventory: list[int]) -> None:
        self.collection = _FakeRainCollection(latest_ms, inventory)
        self.collection_id = ""

    def ImageCollection(self, collection_id: str) -> _FakeRainCollection:
        self.collection_id = collection_id
        return self.collection

    def Date(self, value: Any) -> _FakeRainDate:
        return _FakeRainDate(value)


def test_rainfall_fetch_checks_hourly_inventory_before_sampling(monkeypatch: Any) -> None:
    latest = pd.Timestamp.now(tz="UTC").floor("h")
    inventory = (pd.date_range(end=latest, periods=720, freq="h").asi8 // 1_000_000).tolist()
    fake_ee = _FakeRainEE(inventory[-1], inventory)
    monkeypatch.setattr(gee_connector, "initialize_earth_engine", lambda project, credentials: fake_ee)
    monkeypatch.setattr(gee_connector, "_point_collection", lambda ee, records: "points")
    monkeypatch.setattr(gee_connector, "_features_to_frame", lambda ee, sampled: _valid_rainfall_frame())

    result, timestamp = gee_connector._rainfall_features.__wrapped__(
        ((1, 16.8, 96.1), (2, 16.9, 96.2)),
        "test-project",
        None,
    )

    assert fake_ee.collection_id == gee_connector.GSMAP_COLLECTION_ID
    assert fake_ee.collection.selected_band == gee_connector.GSMAP_BAND
    assert result.equals(_valid_rainfall_frame())
    assert pd.Timestamp(timestamp) == latest
