import streamlit as st
import pandas as pd
import joblib

# Load saved files
prod_files = joblib.load('model_prod_files13.pkl')

model = prod_files['model']
OHE = prod_files['cat_encod']
scaler = prod_files['num_encod']

# Title
st.title('🏠 House Price Prediction App')
st.write('Enter House Details')

# Numerical Inputs
area = st.number_input('Area', min_value=500, max_value=20000, value=1500)
bedrooms = st.slider('Bedrooms', 1, 10, 3)
bathrooms = st.slider('Bathrooms', 1, 10, 2)
stories = st.slider('Stories', 1, 5, 2)
parking = st.slider('Parking', 0, 5, 1)

# Categorical Inputs
mainroad = st.selectbox('Main Road', ['yes', 'no'])
guestroom = st.selectbox('Guest Room', ['yes', 'no'])
basement = st.selectbox('Basement', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
furnishingstatus = st.selectbox(
    'Furnishing Status',
    ['furnished', 'semi-furnished', 'unfurnished']
)

# Create DataFrame
input_df = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

# Separate categorical and numerical columns
cat_df = input_df.select_dtypes(include='object')
num_df = input_df.select_dtypes(include='number')

# Transform categorical data
cat_trans = OHE.transform(cat_df)

# Transform numerical data
num_trans = scaler.transform(num_df)

# Combine transformed data
final_data = pd.concat([num_trans, cat_trans], axis=1)

# Prediction Button
if st.button('Predict Price'):

    prediction = model.predict(final_data)

    st.success(f'Estimated House Price: ₹ {prediction[0][0]:,.2f}')
