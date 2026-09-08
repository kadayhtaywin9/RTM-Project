from models import (
    CompoundModel,
    CycloneModel,
    EarthquakeModel,
    FloodModel,
    TowerRecommendationModel,
)


def test_packaged_model_feature_contracts() -> None:
    models = [TowerRecommendationModel(), FloodModel(), CycloneModel(), EarthquakeModel(), CompoundModel()]
    assert all(model.model_path.is_file() for model in models)
    assert all(model.features for model in models)

