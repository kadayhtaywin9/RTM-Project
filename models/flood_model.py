from ._base import TreeModel


class FloodModel(TreeModel):
    def __init__(self) -> None:
        super().__init__("hazard_flood_xgb.json", "hazard_model_metadata.json")

