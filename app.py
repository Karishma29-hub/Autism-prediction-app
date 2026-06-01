import streamlit as st
import numpy as np
import pickle
import os

# ---------------------------
# Load saved files safely
# ---------------------------
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Autism Prediction App")
st.write("Enter the details below:")

# ---------------------------
# Inputs (A1 - A10 + Age)
# ---------------------------
a1 = st.number_input("A1 Score", 0, 1)
a2 = st.number_input("A2 Score", 0, 1)
a3 = st.number_input("A3 Score", 0, 1)
a4 = st.number_input("A4 Score", 0, 1)
a5 = st.number_input("A5 Score", 0, 1)
a6 = st.number_input("A6 Score", 0, 1)
a7 = st.number_input("A7 Score", 0, 1)
a8 = st.number_input("A8 Score", 0, 1)
a9 = st.number_input("A9 Score", 0, 1)
a10 = st.number_input("A10 Score", 0, 1)
age = st.number_input("Age", 1, 100)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict"):

    # Convert input into array
    input_data = np.array([[a1, a2, a3, a4, a5,
                            a6, a7, a8, a9, a10,
                            age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Convert to original label
    label = label_encoder.inverse_transform([prediction])[0]

    # ---------------------------
    # FINAL DISPLAY FIX
    # ---------------------------
    if str(label).lower() in ["autism", "yes", "1"]:
        st.error("❌ Autism Detected")
    else:
        st.success("✅ No Autism Detected")
