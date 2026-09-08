from ._base import TreeModel


class CompoundModel(TreeModel):
    def __init__(self) -> None:
        super().__init__("hazard_compound_xgb.json", "multi_hazard_model_metadata.json", ("hazards", "compound"))

