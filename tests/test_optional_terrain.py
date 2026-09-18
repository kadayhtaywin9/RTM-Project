"""Optional terrain must not turn an OS DLL block into a dashboard outage."""
from __future__ import annotations

import ast
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import site_ai


@pytest.fixture(autouse=True)
def clear_optional_reader_cache():
    site_ai._rasterio_runtime.cache_clear()
    yield
    site_ai._rasterio_runtime.cache_clear()


def test_import_does_not_require_or_attempt_rasterio():
    code = '''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "rasterio" or name.startswith("rasterio."):
        raise AssertionError("Rasterio must not be imported at dashboard startup")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import site_ai
assert site_ai.model_status()["training_rows"] > 0
print("Optional terrain import isolation passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=site_ai.BASE,
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", [
    ImportError("DLL load failed: An Application Control policy has blocked this file"),
    ModuleNotFoundError("No module named rasterio"),
    OSError("Blocked DLL"),
])
def test_unavailable_rasterio_is_cached_and_returns_no_sample(monkeypatch, failure):
    calls = []

    def blocked(name):
        calls.append(name)
        raise failure

    monkeypatch.setattr(site_ai.importlib, "import_module", blocked)
    assert site_ai._sample_elevation(16.86, 96.20) is None
    assert site_ai._sample_elevation(16.85, 96.21) is None
    assert calls == ["rasterio"]


class FakeRaster:
    crs = "EPSG:4326"
    bounds = SimpleNamespace(left=95, right=97, bottom=16, top=18)

    def __init__(self, value, nodata=None):
        self.value, self.nodata = value, nodata

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self, coordinates, masked=False):
        assert masked
        assert coordinates == [(96.2, 16.86)]
        masked_value = np.ma.is_masked(self.value)
        yield np.ma.array([0 if masked_value else self.value], mask=[masked_value])


def fake_reader(monkeypatch, value=12.5, nodata=None):
    reader = SimpleNamespace(
        open=lambda _: FakeRaster(value, nodata),
        errors=SimpleNamespace(RasterioError=RuntimeError),
    )
    monkeypatch.setattr(site_ai, "_rasterio_runtime", lambda: reader)
    return reader


@pytest.mark.parametrize("value", [0.0, -2.0, 12.5])
def test_valid_terrain_is_retained_including_true_zero(monkeypatch, value):
    fake_reader(monkeypatch, value)
    assert site_ai._sample_elevation(16.86, 96.2) == value


@pytest.mark.parametrize("value,nodata", [
    (-9999, -9999), (float("nan"), None), (float("inf"), None), (np.ma.masked, None),
])
def test_invalid_terrain_is_unknown_not_zero(monkeypatch, value, nodata):
    fake_reader(monkeypatch, value, nodata)
    assert site_ai._sample_elevation(16.86, 96.2) is None


def test_outside_raster_does_not_sample_a_fabricated_zero(monkeypatch):
    fake_reader(monkeypatch, 0)
    assert site_ai._sample_elevation(20.0, 100.0) is None


@pytest.mark.parametrize("failure", [ImportError("DLL blocked"), OSError("Missing TIFF"), RuntimeError("Raster read error")])
def test_open_failure_is_nonfatal(monkeypatch, failure):
    reader = fake_reader(monkeypatch)

    def fail(_):
        raise failure

    reader.open = fail
    assert site_ai._sample_elevation(16.86, 96.2) is None


def test_site_score_retains_proxy_provenance_in_report(monkeypatch):
    monkeypatch.setattr(site_ai, "_rasterio_runtime", lambda: None)
    result = site_ai.assess_site(16.86, 96.2)
    assert result["ok"]
    assert result["elevation_is_proxy"]
    assert result["elevation_m"] is not None
    assert "stored area-candidate" in result["terrain_warning"]
    assert "Application Control" in result["terrain_warning"]
    assert "exact" not in result["feature_sources"]["elevation_score"]
    # Execute only the report helper; importing app.py would run the entire UI.
    tree = ast.parse((site_ai.BASE / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_ai_report_text")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)  # noqa: S102 - trusted local report helper only
    report = namespace["_ai_report_text"](result)
    assert result["elevation_source"] in report
    assert result["terrain_warning"] in report
    assert "tower gap and elevation are evaluated at the selected coordinate" not in report


def test_available_sample_is_not_labelled_as_proxy(monkeypatch):
    monkeypatch.setattr(site_ai, "_sample_elevation", lambda *_: 12.5)
    result = site_ai.extract_site_features(16.86, 96.2)
    assert result["ok"] and not result["elevation_is_proxy"]
    assert result["terrain_warning"] is None
    assert result["elevation_m"] == 12.5
    assert "GeoTIFF pixel" in result["elevation_source"]


def test_no_valid_sample_or_proxy_refuses_score(monkeypatch):
    original = site_ai._area_proxy_candidate

    def invalid_proxy(*args):
        proxy = original(*args).copy()
        proxy["elevation_m"] = np.nan
        proxy["elevation_score"] = np.nan
        return proxy

    monkeypatch.setattr(site_ai, "_rasterio_runtime", lambda: None)
    monkeypatch.setattr(site_ai, "_area_proxy_candidate", invalid_proxy)
    result = site_ai.assess_site(16.86, 96.2)
    assert not result["ok"]
    assert "Missing terrain is not treated as zero" in result["reason"]
    assert "score_100" not in result


def test_outside_study_area_still_refuses_score(monkeypatch):
    monkeypatch.setattr(site_ai, "_rasterio_runtime", lambda: None)
    result = site_ai.assess_site(0, 0)
    assert not result["ok"] and "outside" in result["reason"]
