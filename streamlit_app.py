# Program developed by Madanjit Kumar Singh
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# streamlit_app.py

# Import libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scaler, and feature info
model_data = joblib.load("house_price_model_GB.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
numeric_cols = model_data["numeric_cols"]
feature_cols = model_data["feature_cols"]

# Frequent sellers used during training
top_sellers = ['Jellis', 'Biggin', 'Nelson', 'Buxton', 'Kay', 'Zed', 'Raywhite',
               'Marshall', 'Hocking', 'Other']

# --- Streamlit UI ---
st.title("🏠 Melbourne House Price Predictor")
st.caption("Developed by: Ravindra Singh")

st.markdown("Enter property details to get the estimated price prediction:")

# Main 3-column layout
col1, col2, col3 = st.columns(3, gap="small")

with col1:
    # Property Basics
    rooms = st.number_input("Rooms:", 1, 10, 2)
    bedrooms = st.number_input("Bedrooms:", 1, 10, 2)
    bathrooms = st.number_input("Bathrooms:", 1, 5, 1)
    car = st.number_input("Car Spaces:", 0, 5, 1)

with col2:
    # Location
    suburb = st.selectbox("Suburb:", ["Richmond", "Brighton", "Preston"])
    distance = st.number_input("Distance from CBD (KM):", 0.0, 50.0, 5.0, step=0.5)
    landsize = st.number_input("Land Size:", 0.0, 2000.0, 100.0, step=10.0)
    prop_type = st.selectbox("Property Type:", ['h', 'u', 't', 'a'], format_func=lambda x:
                           {'h':'House', 'u':'Unit', 't':'Townhouse', 'a':'Apartment'}[x])

with col3:
    # Sale Info
    year = st.selectbox("Sold Year:", [2023, 2024, 2025])
    month = st.selectbox("Sold Month:", list(range(1, 13)))
    price_per_sqm = st.number_input("Price/sqm:", 0.0, 20000.0, 3000.0, step=100.0)
    postcode = st.text_input("Postcode:", "3121")

# Advanced options in expander
with st.expander("Advanced Options:", expanded=False):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        latitude = st.number_input("Latitude:", value=-37.81, format="%.5f")
        schools_nearby = st.number_input("Schools Nearby:", 0, 10, 2)
    with adv_col2:
        longitude = st.number_input("Longitude:", value=144.96, format="%.5f")
        council = st.text_input("Council:", "Yarra")
    with adv_col3:
        seller = st.text_input("Seller:", "Jellis Craig")
        region = st.selectbox("Region:", ['Inner Melbourne', 'Inner South', 'Melbourne - North', 'Yarra'])

# Predict button centered
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Predict Price", type="primary"):
    # --- Construct input data ---
    user_input = {
        'Rooms': rooms,
        'Distance': distance,
        'Bedroom': bedrooms,
        'Bathroom': bathrooms,
        'Car': car,
        'Landsize': landsize,
        'Latitude': latitude,
        'Longitude': longitude,
        'schools_nearby': schools_nearby,
        'PricePerSqMeter': price_per_sqm,
        'Suburb': suburb,
        'Type': prop_type,
        'SellerG': seller if seller in top_sellers else 'Other',
        'CouncilArea': council,
        'Regionname': region,
        'Postcode': postcode,
        'SaleYear': year,
        'SaleMonth': month
    }

    df = pd.DataFrame([user_input])

    # --- One-hot encode like training ---
    categorical_cols = ['Suburb', 'Type', 'SellerG', 'CouncilArea', 'Regionname', 'Postcode']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Add missing one-hot columns and order
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_cols]

    # --- Scale numeric features ---
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # --- Predict ---
    prediction = model.predict(df_encoded)[0]
    
    st.success(f"💰 Estimated Price: ${prediction:,.2f}")
st.markdown("</div>", unsafe_allow_html=True)
