from ._base import TreeModel


class EarthquakeModel(TreeModel):
    def __init__(self) -> None:
        super().__init__("hazard_earthquake_xgb.json", "multi_hazard_model_metadata.json", ("hazards", "earthquake"))

