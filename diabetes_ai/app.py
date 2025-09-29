import streamlit as st
import numpy as np
import joblib
import os

# ------------------------------
# Set base directory to the folder containing app.py
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model and scaler
MODEL_PATH = os.path.join(BASE_DIR, "model_lr.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# Load trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------
# Streamlit App
# ------------------------------
st.title("Prédiction du risque de diabète avec l’IA")
st.title("Cours ADM3763 A00")

# Input fields
preg = st.number_input("Grossesses", 0, 20, 1)
gluc = st.number_input("Glucose", 0, 300, 120)
bp   = st.number_input("Tension artérielle", 0, 200, 70)
skin = st.number_input("Épaisseur de la peau", 0, 100, 20)
ins  = st.number_input("Insuline", 0, 900, 80)
bmi  = st.number_input("IMC (Indice de Masse Corporelle)", 0.0, 70.0, 25.0)
dpf  = st.number_input("Fonction de prédisposition au diabète", 0.0, 3.0, 0.5)
age  = st.number_input("Âge", 0, 120, 30)

# Predict button
if st.button("Prédire"):
    # Create feature array and scale
    X = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
    X_scaled = scaler.transform(X)
    
    # Get prediction and probability
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]
    
    # Show probability
    st.write(f"Estimated Probability: {proba:.2f}")
    
    # Show risk status
    if pred == 1:
        st.success("À risque")
    else:

        st.info("Pas à risque")

