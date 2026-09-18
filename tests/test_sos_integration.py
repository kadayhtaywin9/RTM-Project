import json
from pathlib import Path

import pandas as pd

from utils.sos import enrich_sos_incident, nearest_tower, resolve_township


BASE = Path(__file__).resolve().parents[1]


def test_sos_point_resolves_to_township_and_nearest_tower():
    admin3 = json.loads((BASE / "data" / "yangon_admin3.geojson").read_text(encoding="utf-8"))
    towers = pd.read_csv(BASE / "data" / "tower_sites_yangon_all.csv")
    sample = {"id": "SOS-TEST", "latitude": 16.77544, "longitude": 96.13243, "accuracy_m": 10.0}

    assert resolve_township(sample["latitude"], sample["longitude"], admin3) == "Ahlone"
    tower = nearest_tower(sample["latitude"], sample["longitude"], towers)
    assert tower is not None
    assert tower["adm3_name"] == "Ahlone"
    assert tower["distance_km"] < 0.01

    enriched = enrich_sos_incident(sample, admin3, towers)
    assert enriched["township"] == "Ahlone"
    assert enriched["nearest_tower_km"] < 0.01
