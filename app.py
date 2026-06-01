import streamlit as st
import pandas as pd
import pickle

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("Autism Prediction System")

# ---------------- INPUT SECTION ---------------- #

scores = {}
for i in range(1, 11):
    scores[f"A{i}_Score"] = st.selectbox(
        f"A{i} Score",
        [0, 1]
    )

age = st.number_input("Age", min_value=1, max_value=100)

gender = st.selectbox("Gender", ['f', 'm'])

ethnicity = st.selectbox(
    "Ethnicity",
    ['Asian', 'Black', 'Hispanic', 'Latino',
     'Middle Eastern ', 'Others', 'Pasifika',
     'South Asian', 'Turkish', 'White-European']
)

jaundice = st.selectbox("Jaundice", ['no', 'yes'])

# ⚠️ FIXED SPELLING (austim from your encoder)
austim = st.selectbox(
    "Family Member with Autism",
    ['no', 'yes']
)

country = st.selectbox(
    "Country",
    list(encoders['contry_of_res'].classes_)
)

used_app_before = st.selectbox(
    "Used App Before",
    ['no', 'yes']
)

relation = st.selectbox(
    "Relation",
    ['Others', 'Self']
)

result = st.number_input(
    "Screening Result",
    min_value=0.0
)

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    input_data = np.array([[a1, a2, a3, a4, a5,
                            a6, a7, a8, a9, a10,
                            age]])

    # SCALE HERE (IMPORTANT)
    input_scaled = scaler.transform(input_data)

    # PREDICT HERE
    prediction = model.predict(input_scaled)[0]

    label = label_encoder.inverse_transform([prediction])[0]

    if str(label).lower() in ["autism", "yes", "1"]:
        st.error("❌ Autism Traits Detected")
    else:
        st.success("✅ No Autism Traits Detected")
