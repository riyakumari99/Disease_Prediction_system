import streamlit as st
import requests

st.title("Disease Prediction System 🏥")

symptoms = st.text_input("Enter symptoms (comma separated)")

if st.button("Predict"):
    symptom_list = [s.strip() for s in symptoms.split(",")]

    response = requests.get(
        "http://127.0.0.1:5000/predict_get",
        params={"symptoms": ",".join(symptom_list)}
    )

    result = response.json()

    st.success(f"Disease: {result['disease']}")
    st.write("Description:", result["description"])
    st.write("Precautions:", result["precaution"])