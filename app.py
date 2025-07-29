import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the scaler and model
scaler = joblib.load("scaler.pkl")  # Load scaler
model = joblib.load("model.pkl")    # Load model

st.title("Churn Prediction App")
st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")
st.divider()

# Input fields
user_name = st.text_input("Enter your Name")
phone_number = st.text_input("Enter your Phone Number (10 digits)")

age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charges", min_value=30, max_value=150)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

# Button to trigger prediction
if st.button("Predict!"):
    # Validate Name and Phone Number
    if not user_name.strip():
        st.warning("Please enter your Name!")
    elif not phone_number.isdigit() or len(phone_number) != 10:
        st.warning("Please enter a valid 10-digit Phone Number!")
    else:
        # Convert gender to numeric
        gender_selected = 1 if gender == "Female" else 0

        # Create input data
        x = [age, gender_selected, tenure, monthlycharge]
        
        # Convert input into a DataFrame with column names
        feature_names = ["Age", "Gender", "Tenure", "MonthlyCharges"]
        X1 = pd.DataFrame([x], columns=feature_names)
        
        # Apply transformation
        X_array = scaler.transform(X1)
        
        # Make prediction
        prediction = model.predict(X_array)[0]
        prediction = "Yes" if prediction == 1 else "No"
        
        st.balloons()
        st.success(f"Hello {user_name}, the churn prediction for the given details is: **{prediction}**")
        st.write(f"📞 Contact: {phone_number}")
        # to run this code type---> streamlit run app.py in new terminal
