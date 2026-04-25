import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open('model_hpp.pkl', 'rb'))
scaler = pickle.load(open('scaler_hpp.pkl', 'rb'))

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏠 Smart House Price Predictor</h1>", unsafe_allow_html=True)

st.markdown("### ✨ Fill in the details below to estimate your property value")

st.write("---")

# Sidebar Inputs
st.sidebar.header("🏡 Property Details")

square_footage = st.sidebar.selectbox('Square Footage', list(range(500, 5001, 100)))
num_bedrooms = st.sidebar.selectbox('Bedrooms', [1, 2, 3, 4, 5])
num_bathrooms = st.sidebar.selectbox('Bathrooms', [1, 2, 3])
year_built = st.sidebar.selectbox('Year Built', list(range(1950, 2024)))
lot_size = st.sidebar.selectbox('Lot Size (acres)', [round(i,1) for i in np.arange(0.5, 5.1, 0.5)])
garage_size = st.sidebar.selectbox('Garage Size', [0, 1, 2])
neighborhood_quality = st.sidebar.selectbox('Neighborhood Quality', list(range(1, 11)))

# Button
st.write("### 🔍 Get Your Price Prediction")

if st.button('💰 Predict Price'):

    input_data = pd.DataFrame([[square_footage, num_bedrooms, num_bathrooms,
                                year_built, lot_size, garage_size, neighborhood_quality]],
                              columns=['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms',
                                       'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality'])

    scaled_input = scaler.transform(input_data)
    log_pred = model.predict(scaled_input)
    predicted_price = np.exp(log_pred[0])

    st.success(f"🏡 Estimated Property Value: ₹ {predicted_price:,.2f}")

    st.balloons()