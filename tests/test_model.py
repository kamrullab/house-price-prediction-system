import pandas as pd
import pytest

from generate_sample_data import generate
from src.model import FEATURES, compare_models, train_and_evaluate, validate_data


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


def test_model_comparison_returns_both_models():
    results = compare_models(generate(rows=120))
    assert set(results["model"]) == {"Linear Regression", "Random Forest"}
    assert results["r2"].between(-1, 1).all()


def test_unknown_location_can_be_predicted():
    model, _, _ = train_and_evaluate(generate(rows=100))
    sample = pd.DataFrame([[1500, 3, 2, 5, "Unknown ZIP"]], columns=FEATURES)
    assert model.predict(sample).shape == (1,)


def test_numeric_features_are_standardized():
    model, _, _ = train_and_evaluate(generate(rows=100))
    numeric_pipeline = model.named_steps["preprocessing"].named_transformers_["numeric"]
    assert "scaler" in numeric_pipeline.named_steps
    assert numeric_pipeline.named_steps["scaler"].mean_.shape == (4,)

