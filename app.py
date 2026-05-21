import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction System")

st.write("Enter patient medical information")

# User Inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 300)
blood_pressure = st.number_input("Blood Pressure", 0, 200)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Prediction
if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    risk = probability[0][1] * 100

    if prediction[0] == 1:
        st.error("High Diabetes Risk")
    else:
        st.success("Low Diabetes Risk")

    st.write(f"Probability of Diabetes: {risk:.2f}%")
