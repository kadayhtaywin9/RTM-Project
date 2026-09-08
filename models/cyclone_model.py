from ._base import TreeModel


class CycloneModel(TreeModel):
    def __init__(self) -> None:
        super().__init__("hazard_cyclone_xgb.json", "multi_hazard_model_metadata.json", ("hazards", "cyclone"))

