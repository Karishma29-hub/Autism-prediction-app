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

    # Encode categorical values (FIXED KEY NAME HERE)
    gender_enc = encoders['gender'].transform([gender])[0]
    ethnicity_enc = encoders['ethnicity'].transform([ethnicity])[0]
    jaundice_enc = encoders['jaundice'].transform([jaundice])[0]
    austim_enc = encoders['austim'].transform([austim])[0]
    country_enc = encoders['contry_of_res'].transform([country])[0]
    used_app_enc = encoders['used_app_before'].transform([used_app_before])[0]
    relation_enc = encoders['relation'].transform([relation])[0]

    # Create input dataframe
    input_df = pd.DataFrame([{
        'A1_Score': scores['A1_Score'],
        'A2_Score': scores['A2_Score'],
        'A3_Score': scores['A3_Score'],
        'A4_Score': scores['A4_Score'],
        'A5_Score': scores['A5_Score'],
        'A6_Score': scores['A6_Score'],
        'A7_Score': scores['A7_Score'],
        'A8_Score': scores['A8_Score'],
        'A9_Score': scores['A9_Score'],
        'A10_Score': scores['A10_Score'],
        'age': age,
        'gender': gender_enc,
        'ethnicity': ethnicity_enc,
        'jaundice': jaundice_enc,
        'austim': austim_enc,
        'country_of_res': country_enc,
        'used_app_before': used_app_enc,
        'relation': relation_enc
    }])

    # DEBUG INFO (VERY IMPORTANT)
    st.write("Model expects features:", model.n_features_in_)
    st.write("Input shape:", input_df.shape)
    st.write("Columns:", list(input_df.columns))

    # Prediction
    prediction = model.predict(input_df)

    st.write("Prediction value:", prediction[0])

    # RESULT LOGIC (IMPORTANT: CONFIRM 0/1 MEANING IN YOUR MODEL)
    if prediction[0] == 1:
        st.error("Autism Traits Detected")
    else:
        st.success("No Autism Traits Detected")
