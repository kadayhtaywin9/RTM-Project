"""Cesium maps with the existing dashboard layer and selection contracts."""
from __future__ import annotations

import hashlib
import base64
import json
import math
import os
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import shape
from streamlit.errors import StreamlitSecretNotFoundError


_component = components.declare_component("rtm_cesium", path=str(Path(__file__).parent / "cesium_frontend"))


def _plain(value):
    if isinstance(value, dict):
        if "bdata" in value and "dtype" in value:
            array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
            if value.get("shape"):
                array = array.reshape(tuple(int(x.strip()) for x in value["shape"].split(",")))
            return _plain(array.tolist())
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def ion_token() -> str:
    try:
        return str(st.secrets.get("CESIUM_ION_TOKEN", os.getenv("CESIUM_ION_TOKEN", ""))).strip()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return os.getenv("CESIUM_ION_TOKEN", "").strip()


def figure_payload(fig, *, pick_location=False) -> dict:
    """Serialize original values; avoid Plotly's binary-array JSON encoding."""
    raw = _plain(fig.to_plotly_json())
    layout = raw.get("layout", {})
    map_layout = layout.get("map", layout.get("mapbox", {}))
    layers = []
    for trace in raw.get("data", []):
        if trace.get("type") not in {"scattermap", "choroplethmap", "scattermapbox", "choroplethmapbox"}:
            raise ValueError("Cesium accepts map traces only.")
        layers.append({k: v for k, v in trace.items() if k in {
            "type", "name", "lat", "lon", "mode", "marker", "line", "text", "customdata",
            "hovertemplate", "hoverinfo", "geojson", "locations", "z", "zmin", "zmax",
            "colorscale", "showscale", "colorbar", "featureidkey", "visible", "fill", "fillcolor",
        }})
    scene = {"layers": layers, "center": map_layout.get("center", {"lat": 16.86, "lon": 96.20}),
             "zoom": map_layout.get("zoom", 9), "height": int(layout.get("height", 520)),
             "pick_location": bool(pick_location)}
    for layer in layers:
        if layer.get("name") == "Study area boundary" and layer.get("geojson", {}).get("features"):
            bounds = [shape(f["geometry"]).bounds for f in layer["geojson"]["features"]]
            scene["bounds"] = [min(b[0] for b in bounds), min(b[1] for b in bounds),
                               max(b[2] for b in bounds), max(b[3] for b in bounds)]
            break
    if "bounds" not in scene:
        # Detail and SOS maps fit their actual locations in either projection.
        coordinates = [(float(lon), float(lat)) for layer in layers
                       for lat, lon in zip(layer.get("lat", []), layer.get("lon", []))
                       if lat is not None and lon is not None
                       and math.isfinite(float(lat)) and math.isfinite(float(lon))
                       and -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180]
        if coordinates:
            lons, lats = zip(*coordinates)
            scene["bounds"] = [min(lons), min(lats), max(lons), max(lats)]
    scene["revision"] = hashlib.sha256(json.dumps(scene, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    return scene


def selection_from_event(event, scene, *, selectable=False):
    empty = {"selection": {"points": []}}
    if not isinstance(event, dict) or event.get("revision") != scene["revision"]:
        return empty
    if scene["pick_location"] and isinstance(event.get("location"), dict):
        try:
            lat, lon = float(event["location"]["lat"]), float(event["location"]["lng"])
            if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                return {**empty, "last_clicked": {"lat": lat, "lng": lon}}
        except (TypeError, ValueError, KeyError):
            pass
    if selectable:
        try:
            layer_index, point_index = int(event["layer"]), int(event["point"])
            if layer_index < 0 or point_index < 0:
                return empty
            layer = scene["layers"][layer_index]
            point = {"curve_number": layer_index, "point_index": point_index,
                     "lat": layer["lat"][point_index], "lon": layer["lon"][point_index]}
            if "customdata" in layer:
                point["customdata"] = layer["customdata"][point_index]
            return {"selection": {"points": [point]}}
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return empty


def render_map(fig, *, key=None, on_select=None, pick_location=False, **_):
    scene = figure_payload(fig, pick_location=pick_location)
    event = _component(scene=scene, token=ion_token(), selectable=on_select == "rerun",
                       key=key or "cesium_" + scene["revision"], default=None)
    return selection_from_event(event, scene, selectable=on_select == "rerun")
