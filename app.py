
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("Melbourne House Price Prediction")

# Load dataset
df = pd.read_excel("regression.xlsx")

# Fill missing values
df["parking"] = df["parking"].fillna(df["parking"].median())
df["land_size"] = df["land_size"].fillna(df["land_size"].median())

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=["suburb", "property_type"],
    drop_first=True
)

# Feature engineering
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

# Date processing
df["sale_year"] = pd.to_datetime(df["sale_date"]).dt.year
df["sale_month"] = pd.to_datetime(df["sale_date"]).dt.month

df = df.drop("sale_date", axis=1)

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

st.header("Enter Property Details")

bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
parking = st.number_input("Parking", 0, 10, 1)
land_size = st.number_input("Land Size", 50, 1000, 300)

# Create input dataframe
input_data = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "parking": [parking],
    "land_size": [land_size],
    "total_rooms": [bedrooms + bathrooms],
    "sale_year": [2025],
    "sale_month": [5]
})

# Add missing columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Match column order
input_data = input_data[X.columns]

# Prediction
prediction = model.predict(input_data)

st.subheader("Predicted House Price")
st.write(f"${prediction[0]:,.2f}")
