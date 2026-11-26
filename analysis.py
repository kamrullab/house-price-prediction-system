"""Generate EDA, evaluation charts, and model-comparison results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.model import NUMERIC_FEATURES, compare_models, train_and_evaluate

DATA_PATH = Path("data/housing.csv")
OUTPUT_DIR = Path("reports/figures")


def save_figure(name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    sns.set_theme(style="whitegrid", palette="deep")

    numeric = data[["area", "bedrooms", "bathrooms", "age", "price"]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Feature Correlation Heatmap")
    save_figure("correlation_heatmap.png")

    sample = data.sample(min(5000, len(data)), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="area", y="price", alpha=0.35, s=22)
    plt.title("Living Area and Sale Price")
    plt.xlabel("Living area in square feet")
    plt.ylabel("Sale price in USD")
    save_figure("area_vs_price.png")

    model, metrics, comparison = train_and_evaluate(data)
    comparison["residual"] = comparison["actual"] - comparison["predicted"]

    coefficients = model.named_steps["model"].coef_[: len(NUMERIC_FEATURES)]
    feature_impact = pd.DataFrame(
        {
            "feature": NUMERIC_FEATURES,
            "standardized_coefficient_usd": coefficients,
            "absolute_impact_usd": abs(coefficients),
        }
    ).sort_values("absolute_impact_usd", ascending=False)
    feature_impact.to_csv(OUTPUT_DIR.parent / "standardized_feature_impact.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_impact, x="standardized_coefficient_usd", y="feature", color="#2E6F9E")
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("Standardized Numeric Feature Impact")
    plt.xlabel("Price change in USD for one standard-deviation increase")
    plt.ylabel("Feature")
    save_figure("standardized_feature_impact.png")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=comparison, x="actual", y="predicted", alpha=0.4, s=24)
    bounds = [comparison[["actual", "predicted"]].min().min(), comparison[["actual", "predicted"]].max().max()]
    plt.plot(bounds, bounds, "--", color="#C0392B", linewidth=1.5, label="Perfect prediction")
    plt.title("Actual and Predicted Prices")
    plt.xlabel("Actual price in USD")
    plt.ylabel("Predicted price in USD")
    plt.legend()
    save_figure("actual_vs_predicted.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=comparison, x="predicted", y="residual", alpha=0.4, s=24)
    plt.axhline(0, color="#C0392B", linestyle="--", linewidth=1.5)
    plt.title("Residuals by Predicted Price")
    plt.xlabel("Predicted price in USD")
    plt.ylabel("Residual in USD")
    save_figure("residual_plot.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(comparison["residual"], bins=50, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual in USD")
    save_figure("residual_distribution.png")

    comparison_results = compare_models(data)
    comparison_results.to_csv(OUTPUT_DIR.parent / "model_comparison.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR.parent / "linear_regression_metrics.csv", index=False)
    print(comparison_results.to_string(index=False))
    print(f"Charts and metrics saved under {OUTPUT_DIR.parent}.")


if __name__ == "__main__":
    main()
