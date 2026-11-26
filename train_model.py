"""Train and save the house price prediction model."""

import argparse
from pathlib import Path

import pandas as pd

from src.model import save_model, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the house price model")
    parser.add_argument("--data", default="data/housing.csv", help="Input CSV path")
    parser.add_argument("--output", default="models/house_price_model.joblib", help="Model output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model, metrics, _ = train_and_evaluate(data)
    save_model(model, Path(args.output))
    print(f"Model saved to {args.output}")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R2:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

