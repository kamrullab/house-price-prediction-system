"""Generate a reproducible demonstration housing dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = np.array(["City Center", "Suburb", "Residential", "Outskirts"])
    location = rng.choice(locations, size=rows, p=[0.2, 0.3, 0.35, 0.15])
    area = rng.integers(550, 3500, size=rows)
    bedrooms = np.clip(np.rint(area / 650 + rng.normal(0, 0.7, rows)), 1, 7).astype(int)
    bathrooms = np.clip(np.rint(bedrooms / 2 + rng.normal(0.5, 0.5, rows)), 1, 5).astype(int)
    age = rng.integers(0, 45, size=rows)
    location_value = pd.Series(location).map(
        {"City Center": 3_000_000, "Suburb": 1_500_000, "Residential": 900_000, "Outskirts": 0}
    ).to_numpy()
    price = (
        1_000_000
        + area * 5_000
        + bedrooms * 300_000
        + bathrooms * 450_000
        - age * 35_000
        + location_value
        + rng.normal(0, 600_000, rows)
    )
    return pd.DataFrame(
        {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age,
            "location": location,
            "price": np.maximum(price, 500_000).round(0).astype(int),
        }
    )


if __name__ == "__main__":
    output = Path("data/housing.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(output, index=False)
    print(f"Created {output} with 500 demonstration records.")

