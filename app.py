import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Load model + preprocessing
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------------------------
# App UI
# ---------------------------
st.title("Autism Prediction App")

st.write("Enter patient details below:")

# Example inputs (adjust based on your dataset)
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
age = st.number_input("Age")

# ---------------------------
# Prediction button
# ---------------------------
if st.button("Predict"):

    # Combine input into array
    input_data = np.array([[a1, a2, a3, a4, a5,
                            a6, a7, a8, a9, a10,
                            age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict (IMPORTANT: always take first value only)
    prediction = model.predict(input_scaled)[0]

    # ---------------------------
    # FIX: remove 1,19 issue
    # ---------------------------

    # Ensure prediction is integer
    prediction = int(prediction)

    # Convert to readable label
    result = label_encoder.inverse_transform([prediction])[0]

    # Final output
    st.success(f"Prediction Result: {result}")
