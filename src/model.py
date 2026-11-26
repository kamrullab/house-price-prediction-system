"""Training and inference utilities for the house price model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "age"]
CATEGORICAL_FEATURES = ["location"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_data(data: pd.DataFrame) -> None:
    """Raise a helpful error when a required column is missing."""
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(data) < 10:
        raise ValueError("At least 10 rows are required to train and evaluate the model.")


def build_pipeline() -> Pipeline:
    """Create a preprocessing and multiple-linear-regression pipeline."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])


def train_and_evaluate(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train the pipeline and return it together with evaluation results."""
    validate_data(data)
    clean = data.drop_duplicates().copy()
    clean = clean.dropna(subset=[TARGET])
    clean[CATEGORICAL_FEATURES] = clean[CATEGORICAL_FEATURES].astype("string")
    if len(clean) < 10:
        raise ValueError("At least 10 rows with a price are required after cleaning.")

    x_train, x_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=test_size, random_state=random_state
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    comparison = pd.DataFrame({"actual": y_test, "predicted": predictions}).reset_index(drop=True)
    return pipeline, metrics, comparison


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """Persist a trained pipeline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted pipeline."""
    return joblib.load(path)

