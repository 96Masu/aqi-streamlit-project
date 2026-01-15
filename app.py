import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌍 Air Quality Index (AQI) Prediction")
st.write("Semester 6 Machine Learning Project")

@st.cache_data
def load_data():
    return pd.read_csv("Air_Quality.csv")

df = load_data()
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(columns=["Date", "City", "CO2", "AQI"])
y = df["AQI"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

st.sidebar.header("Enter Pollution Values")

co = st.sidebar.number_input("CO", 0.0, 500.0, 10.0)
no2 = st.sidebar.number_input("NO2", 0.0, 200.0, 10.0)
so2 = st.sidebar.number_input("SO2", 0.0, 100.0, 5.0)
o3 = st.sidebar.number_input("O3", 0.0, 200.0, 20.0)
pm25 = st.sidebar.number_input("PM2.5", 0.0, 300.0, 25.0)
pm10 = st.sidebar.number_input("PM10", 0.0, 300.0, 30.0)

if st.button("Predict AQI"):
    input_data = np.array([[co, no2, so2, o3, pm25, pm10]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted AQI: {prediction:.2f}")
