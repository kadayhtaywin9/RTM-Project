"""Check installed runtime requirements and imports without accessing the network."""
from __future__ import annotations

import importlib
import sys
import traceback
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMPORTS = (
    "streamlit",
    "pandas",
    "numpy",
    "plotly.graph_objects",
    "geopandas",
    "rasterio",
    "scipy.spatial",
    "pyproj",
    "shapely",
    "xgboost",
    "sklearn",
    "folium",
    "streamlit_folium",
    "ee",
    "google.auth",
)


def requirement_errors(
    requirements_text: str,
    version_lookup: Callable[[str], str] = metadata.version,
) -> list[str]:
    """Check every requirement, including its bounds, rather than selected imports."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return ["The packaging dependency is missing; runtime packages need installation."]

    errors = []
    for raw_line in requirements_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            errors.append(f"Invalid requirement in requirements.txt: {line}")
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = version_lookup(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing: {requirement}")
            continue
        if not requirement.specifier.contains(installed):
            errors.append(f"Incompatible: {requirement.name} {installed}; required {requirement.specifier}")
    return errors


def main() -> int:
    if sys.version_info[:2] not in {(3, 11), (3, 12)} or sys.maxsize <= 2**32:
        print("GeoVision requires 64-bit Python 3.11 or 3.12.")
        return 1
    requirements_path = PROJECT_ROOT / "requirements.txt"
    try:
        errors = requirement_errors(requirements_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        print(f"Cannot read runtime requirements: {exc}")
        return 1
    if errors:
        print("Runtime dependency check:")
        for error in errors:
            print(f"  - {error}")
        return 1

    failed_imports = []
    for module_name in RUNTIME_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - expose binary and dependency import failures
            failed_imports.append(module_name)
            print(f"\nCannot import {module_name}:", file=sys.stderr)
            traceback.print_exception(exc)
    if failed_imports:
        print("Runtime imports failed: " + ", ".join(failed_imports))
        return 1
    print("All runtime requirements and imports passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
