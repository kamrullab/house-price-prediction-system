# House Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Linear_Regression-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-2_Passed-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-8A2BE2?style=for-the-badge)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-Public-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kamrullab/house-price-prediction-system)

A complete academic machine-learning project that predicts residential property prices with Multiple Linear Regression. It includes reproducible sample data, preprocessing, model evaluation, automated tests, and an interactive Streamlit web application.

> The bundled CSV contains **synthetic demonstration data**, not real market observations. Replace it with a verified local dataset before using the system for research or decision-making.

## Features

- Predicts prices from area, bedrooms, bathrooms, property age, and location
- Handles missing numeric and categorical values
- Encodes previously unseen locations safely
- Reports MAE, RMSE, and R² on a reproducible 80/20 test split
- Displays actual-versus-predicted results
- Saves the complete preprocessing and model pipeline with Joblib
- Includes unit tests and an interactive Streamlit interface

## Project structure

```text
.
├── app.py                      # Streamlit web application
├── data/housing.csv            # Synthetic demonstration dataset
├── generate_sample_data.py     # Recreates the sample dataset
├── src/model.py                # Validation, preprocessing, training, inference
├── tests/test_model.py         # Automated tests
├── train_model.py              # Command-line training entry point
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Runtime and test dependencies
```

## Requirements

- Python 3.10 or newer
- Git (only needed for cloning)

## Setup on Windows PowerShell

Run each command separately. Every command has its own copy button on GitHub.

```powershell
git clone https://github.com/kamrullab/house-price-prediction-system.git
```

```powershell
cd house-price-prediction-system
```

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
python generate_sample_data.py
```

```powershell
python train_model.py
```

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit, normally <http://localhost:8501>.

If PowerShell prevents activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Setup on macOS or Linux

```bash
git clone https://github.com/kamrullab/house-price-prediction-system.git
cd house-price-prediction-system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_sample_data.py
python train_model.py
streamlit run app.py
```

## Train with your own data

Create a CSV containing these exact columns:

| Column | Type | Example | Meaning |
|---|---:|---:|---|
| `area` | number | `1500` | Floor area in square feet |
| `bedrooms` | integer | `3` | Bedroom count |
| `bathrooms` | integer | `2` | Bathroom count |
| `age` | number | `5` | Property age in years |
| `location` | text | `Suburb` | Area or neighborhood |
| `price` | number | `12500000` | Target price in BDT |

Then train the model:

```bash
python train_model.py --data path/to/your-housing-data.csv --output models/house_price_model.joblib
```

For real data, update the location choices in `app.py` to match the locations used by your dataset.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Methodology

1. Validate required columns and remove duplicate records.
2. Drop records without a target price.
3. Impute missing numeric values with the median.
4. Impute missing locations with the most frequent value and one-hot encode them.
5. Split records into 80% training and 20% testing data with a fixed random seed.
6. Fit a Multiple Linear Regression model.
7. Evaluate predictions with MAE, RMSE, R², and an actual-versus-predicted chart.

The model follows the general form:

```text
price = intercept + b1(area) + b2(bedrooms) + b3(bathrooms)
        + b4(age) + location effects + error
```

## Limitations

- Synthetic data is suitable only for demonstration.
- Housing markets change over time, so real models require recent local data and periodic retraining.
- Linear regression may not capture complex nonlinear relationships.
- A prediction is an estimate, not a certified property valuation.

## License

Released under the [MIT License](LICENSE).

