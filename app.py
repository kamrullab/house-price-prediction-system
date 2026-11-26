"""Streamlit application for training and using the prediction model."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import FEATURES, load_model, save_model, train_and_evaluate

DATA_PATH = Path("data/housing.csv")
MODEL_PATH = Path("models/house_price_model.joblib")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction System")
st.caption("Multiple Linear Regression with reproducible preprocessing and evaluation")


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python generate_sample_data.py` first.")
    model, _, _ = train_and_evaluate(pd.read_csv(DATA_PATH))
    save_model(model, MODEL_PATH)
    return model


predict_tab, evaluate_tab, data_tab = st.tabs(["Predict", "Model evaluation", "Dataset"])

with predict_tab:
    st.subheader("Enter property details")
    left, right = st.columns(2)
    with left:
        area = st.number_input("Area (square feet)", 200, 20_000, 1_500, 50)
        bedrooms = st.number_input("Bedrooms", 1, 20, 3)
        bathrooms = st.number_input("Bathrooms", 1, 20, 2)
    with right:
        age = st.number_input("Property age (years)", 0, 200, 5)
        location = st.selectbox("Location", ["City Center", "Suburb", "Residential", "Outskirts"])

    if st.button("Predict price", type="primary", use_container_width=True):
        row = pd.DataFrame([[area, bedrooms, bathrooms, age, location]], columns=FEATURES)
        prediction = max(0, float(get_model().predict(row)[0]))
        st.metric("Estimated price", f"BDT {prediction:,.0f}")
        st.info("This demonstration estimate is not a professional property valuation.")

with evaluate_tab:
    st.subheader("Holdout evaluation")
    if DATA_PATH.exists():
        _, metrics, comparison = train_and_evaluate(pd.read_csv(DATA_PATH))
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Absolute Error", f"BDT {metrics['mae']:,.0f}")
        c2.metric("Root Mean Squared Error", f"BDT {metrics['rmse']:,.0f}")
        c3.metric("R² score", f"{metrics['r2']:.3f}")
        st.scatter_chart(comparison, x="actual", y="predicted")
    else:
        st.warning("No dataset found. Run `python generate_sample_data.py`.")

with data_tab:
    st.subheader("Training data")
    if DATA_PATH.exists():
        dataset = pd.read_csv(DATA_PATH)
        st.write(f"{len(dataset):,} rows")
        st.dataframe(dataset, use_container_width=True)
    else:
        st.warning("No dataset found.")

