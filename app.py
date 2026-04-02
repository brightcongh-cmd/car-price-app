import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")

# Load dataset
df = pd.read_csv("car_price_prediction.csv")  # replace with your file

# ---- Preprocessing (adapted from your notebook) ----
df['Levy'] = df['Levy'].replace('-', None)
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
df['Levy'] = df['Levy'].fillna(0)

# Features (using Cylinders as requested earlier)
X = df[['Mileage', 'Engine volume', 'Manufacturer', 'Cylinders']]
y = df['Price']

# Encode categorical
X = pd.get_dummies(X, columns=['Manufacturer'], drop_first=True)

# Train model
model = LinearRegression()
# 1. Remove missing values
X = X.dropna()
y = y.loc[X.index]

# 2. Convert categorical columns to numbers
X = pd.get_dummies(X, drop_first=True)

# 3. Ensure all data is numeric
X = X.astype(float)
model.fit(X, y)

# ---- UI ----
st.sidebar.header("Enter Car Details")

mileage = st.sidebar.number_input("Mileage", 0, 500000, 10000)
engine = st.sidebar.number_input("Engine Volume", 0.0, 10.0, 2.0)
cylinders = st.sidebar.number_input("Cylinders", 1, 16, 4)
manufacturer = st.sidebar.selectbox("Manufacturer", df['Manufacturer'].unique())

# Prepare input
input_df = pd.DataFrame({
    'Mileage': [mileage],
    'Engine volume': [engine],
    'Cylinders': [cylinders],
    'Manufacturer': [manufacturer]
})

# Encode input
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: {price:,.2f}")
