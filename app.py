import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle


# Streamlit app
def app():
    st.title("Price Prediction")

    # Get input features from the user
    total_sqft = st.number_input("Total Sq.Ft.", value=0.0, step=1.0)
    bath = st.number_input("Bathrooms", value=0, step=1)
    balcony = st.number_input("Balconies", value=0, step=1)
    bhk = st.number_input("BHK", value=0, step=1)

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({'total_sqft': [total_sqft], 'bath': [bath], 'balcony': [balcony], 'bhk': [bhk]})

    # Load the saved model
    with open('ridge_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Make the prediction
    if st.button("Predict Price"):
        predicted_price = loaded_model.predict(input_data)
        st.write(f"The predicted price is: ${predicted_price[0]:.2f}")

if __name__ == "__main__":
    app()
