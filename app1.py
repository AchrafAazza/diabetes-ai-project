import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = pickle.load(open("diabetes_model.pkl", "rb"))

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to bottom right, #07111f, #0d1b2a);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
}

.metric-card {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.prediction-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.low-risk {
    background-color: rgba(0,255,120,0.15);
    border: 1px solid rgba(0,255,120,0.4);
}

.high-risk {
    background-color: rgba(255,0,80,0.15);
    border: 1px solid rgba(255,0,80,0.4);
}

.small-text {
    color: #b8c7d9;
    font-size: 15px;
}

hr {
    border-color: rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🩺 AI Diabetes Prediction Dashboard")

st.markdown("""
<div class="small-text">
Advanced Machine Learning system for intelligent diabetes risk prediction based on patient medical data.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.title("⚙️ System Information")

    st.markdown("""
    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - NumPy

    ### AI Model
    Random Forest Classifier

    ### Features
    - Real-time prediction
    - Risk analysis
    - Feature importance
    - Interactive dashboard
    """)

# ---------------------------------------------------
# INPUT LAYOUT
# ---------------------------------------------------
st.subheader("📋 Patient Medical Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose Level",
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

with col2:

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

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("🔍 Analyze Patient Risk"):

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

    st.markdown("---")

    st.subheader("🧠 AI Prediction Result")

    if prediction[0] == 1:

        st.markdown(f"""
        <div class="prediction-box high-risk">
        ⚠️ HIGH DIABETES RISK DETECTED
        <br><br>
        Probability: {risk:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.warning("Recommendation: Medical consultation is advised.")

    else:

        st.markdown(f"""
        <div class="prediction-box low-risk">
        ✅ LOW DIABETES RISK
        <br><br>
        Probability: {risk:.2f}%
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(risk))

    # Risk Level
    if risk < 30:
        st.success("Risk Level: LOW")

    elif risk < 70:
        st.warning("Risk Level: MEDIUM")

    else:
        st.error("Risk Level: HIGH")

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
st.markdown("---")

st.subheader("📊 Feature Importance Analysis")

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

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
st.markdown("---")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("AI Model", "Random Forest")

with m2:
    st.metric("Prediction Type", "Binary Classification")

with m3:
    st.metric("Status", "Online")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
<div class="small-text">
Developed using Artificial Intelligence and Machine Learning technologies for healthcare prediction systems.
</div>
""", unsafe_allow_html=True)
