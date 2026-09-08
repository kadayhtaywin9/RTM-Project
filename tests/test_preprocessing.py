import pandas as pd

from data.preprocessing import prepare_tower_features


def test_prepare_tower_features_contract() -> None:
    towers = pd.DataFrame({
        "tower_id": [0, 1],
        "radios": ["GSM", "GSM,LTE"],
        "cell_count": [1, 4],
        "estimated_population_primary_5km": [1000, 5000],
    })
    result = prepare_tower_features(towers)
    assert result["tower_vulnerability"].between(0, 1).all()
    assert result["tower_redundancy"].between(0, 1).all()
    assert result["network_importance"].between(0, 1).all()
    assert result.loc[1, "tower_vulnerability"] < result.loc[0, "tower_vulnerability"]


def test_duplicate_tower_id_is_rejected() -> None:
    towers = pd.DataFrame({"tower_id": [1, 1]})
    try:
        prepare_tower_features(towers)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate tower_id should fail")

