"""Stable orchestration API for GeoVision's multi-hazard system."""
from __future__ import annotations

from typing import Any

import pandas as pd

from data.preprocessing import prepare_tower_features
from hazard_ai import run_hazard_ai as _legacy_hazard_runtime
from utils.caching import cache_resource
from utils.result_context import annotate_result_exports, build_result_context

from .decision_engine import DecisionEngine


class HazardEngine:
    """Coordinate feature preparation, hazard inference, and decisions.

    The established hazard runtime is retained as an adapter during the incremental
    migration, preserving its tested GEE/USGS/local-fallback behavior.
    """

    def __init__(self) -> None:
        self.decisions = DecisionEngine()

    def run(
        self,
        towers: pd.DataFrame,
        *,
        hazard_type: str,
        mode: str = "auto",
        project_id: str | None = None,
        service_account_json: str | dict[str, Any] | None = None,
        rain_date: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        prepared = prepare_tower_features(towers)
        result, run_info = _legacy_hazard_runtime(
            prepared,
            mode=mode,
            project_id=project_id,
            service_account_json=service_account_json,
            rain_date=rain_date,
            hazard_type=hazard_type,
        )
        enriched = self.decisions.enrich(result, hazard_type)
        run_info = dict(run_info)
        run_info["architecture"] = "data connectors → hazard models → hazard engine → decision engine"
        run_info["explainability"] = "Native XGBoost TreeSHAP margin contributions"
        context = build_result_context(enriched, run_info, mode, pd.Timestamp.now(tz="UTC"))
        run_info["result_context"] = context
        return annotate_result_exports(enriched, context), run_info


@cache_resource(max_entries=1)
def get_hazard_engine() -> HazardEngine:
    return HazardEngine()
