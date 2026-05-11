import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('rf_model.pkl', 'rb'))

st.title("Diabetes Predictor")

pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose", 44, 199, 100)
blood_pressure = st.slider("Blood Pressure", 30, 122, 70)
skin_thickness = st.slider("Skin Thickness", 7, 99, 20)
insulin = st.slider("Insulin", 15, 846, 100)
bmi = st.slider("BMI", 18.2, 67.1, 30.0)
dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3)
age = st.slider("Age", 21, 81, 30)

if st.button("Predict"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                                 skin_thickness, insulin, bmi, dpf, age]],
                               columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if result == 1:
        st.error(f"Diabetic — {round(prob*100, 1)}% probability")
    else:
        st.success(f"Not Diabetic — {round(prob*100, 1)}% probability")
