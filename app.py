import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("diabetes_model.pkl", "rb"))

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Diabetes Prediction System")

st.markdown("""
## AI-Based Medical Prediction Application

This intelligent system predicts diabetes risk using Machine Learning algorithms trained on medical data.

The prediction is based on:
- Glucose
- BMI
- Age
- Blood Pressure
- Insulin
- And other medical indicators
""")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("About Project")

st.sidebar.info(
    """
    Intelligent Diabetes Prediction System
    
    Technologies:
    - Python
    - Scikit-learn
    - Streamlit
    - Pandas
    - NumPy
    
    Machine Learning Model:
    Random Forest Classifier
    """
)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Patient Medical Information")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=100
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=79
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict"):

    # Prepare input data
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

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)

    risk = probability[0][1] * 100

    st.subheader("Prediction Result")

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ High Diabetes Risk Detected")
        st.warning("Recommendation: Consult a healthcare professional.")
    else:
        st.success("✅ Low Diabetes Risk")

    # Probability
    st.write(f"### Probability of Diabetes: {risk:.2f}%")

    # Progress bar
    st.progress(int(risk))

    # Risk level
    if risk < 30:
        st.success("Risk Level: LOW")
    elif risk < 70:
        st.warning("Risk Level: MEDIUM")
    else:
        st.error("Risk Level: HIGH")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("Feature Importance")

importance = model.feature_importances_

features = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "DPF",
    "Age"
]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown("""
Developed using Artificial Intelligence and Machine Learning technologies.
""")
