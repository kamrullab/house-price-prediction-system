"""Download and prepare the public King County house-sales dataset."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

OPENML_DATA_ID = 42092
OUTPUT_PATH = Path("data/housing.csv")


def download() -> pd.DataFrame:
    """Fetch OpenML dataset 42092 and map it to the application's schema."""
    dataset = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    source = dataset.frame.copy()

    sale_year = pd.to_datetime(source["date"], errors="coerce").dt.year.fillna(2015)
    prepared = pd.DataFrame(
        {
            "area": pd.to_numeric(source["sqft_living"], errors="coerce"),
            "bedrooms": pd.to_numeric(source["bedrooms"], errors="coerce"),
            "bathrooms": pd.to_numeric(source["bathrooms"], errors="coerce"),
            "age": sale_year - pd.to_numeric(source["yr_built"], errors="coerce"),
            "location": source["zipcode"].astype("string"),
            "price": pd.to_numeric(source["price"], errors="coerce"),
        }
    ).dropna()
    prepared = prepared[(prepared["area"] > 0) & (prepared["price"] > 0) & (prepared["age"] >= 0)]
    prepared["location"] = prepared["location"].str.replace(r"\.0$", "", regex=True)
    return prepared.reset_index(drop=True)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = download()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data):,} real house-sale records to {OUTPUT_PATH}.")

