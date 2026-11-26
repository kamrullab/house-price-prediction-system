import pandas as pd
import pytest

from generate_sample_data import generate
from src.model import FEATURES, train_and_evaluate, validate_data


def test_training_returns_predictions_and_metrics():
    model, metrics, comparison = train_and_evaluate(generate(rows=100))
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert len(comparison) == 20
    assert metrics["r2"] > 0.8
    sample = generate(rows=10)[FEATURES].iloc[[0]]
    assert model.predict(sample).shape == (1,)


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(pd.DataFrame({"price": range(10)}))

