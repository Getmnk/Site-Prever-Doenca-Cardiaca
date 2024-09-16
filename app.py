import streamlit as st
import pandas as pd
import tensorflow as tf

@st.cache_data
def load_model():
    model = tf.keras.models.load_model('heart.pkl')
    return model

with st.spinner("Loading Model...."):
    model = load_model()

def get_user_inputs():
    st.header("Heart Disease Prediction Input (TEST, take it with a grain of salt)")
    
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0, max_value=100)
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol_drinking = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
    stroke = st.selectbox("Have you ever had a stroke?", ["No", "Yes"])
    physical_health = st.number_input("Physical Health (Number of bad physical health days in the last 30 days)", min_value=0, max_value=30, step=1)
    mental_health = st.number_input("Mental Health (Number of bad mental health days in the last 30 days)", min_value=0, max_value=30, step=1)
    diff_walking = st.selectbox("Do you have difficulty walking or climbing stairs?", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    physical_activity = st.selectbox("Do you engage in physical activity?", ["No", "Yes"])
    asthma = st.selectbox("Do you have asthma?", ["No", "Yes"])
    kidney_disease = st.selectbox("Do you have kidney disease?", ["No", "Yes"])
    skin_cancer = st.selectbox("Do you have skin cancer?", ["No", "Yes"])
    
    smoking = 1 if smoking == "Yes" else 0
    alcohol_drinking = 1 if alcohol_drinking == "Yes" else 0
    stroke = 1 if stroke == "Yes" else 0
    diff_walking = 1 if diff_walking == "Yes" else 0
    physical_activity = 1 if physical_activity == "Yes" else 0
    asthma = 1 if asthma == "Yes" else 0
    kidney_disease = 1 if kidney_disease == "Yes" else 0
    skin_cancer = 1 if skin_cancer == "Yes" else 0
    sex = 0 if sex == "Female" else 1 
    
    data = pd.DataFrame({
        "BMI": [bmi],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drinking],
        "Stroke": [stroke],
        "PhysicalHealth": [physical_health],
        "MentalHealth": [mental_health],
        "DiffWalking": [diff_walking],
        "Sex": [sex],
        "PhysicalActivity": [physical_activity],
        "Asthma": [asthma],
        "KidneyDisease": [kidney_disease],
        "SkinCancer": [skin_cancer]
    })

    return data

user_data = get_user_inputs()

if st.button("Predict"):
    predictions = model.predict(user_data)
    
    st.write("Predictions shape:", predictions.shape)
    st.write("Predictions data:", predictions)

    if len(predictions.shape) == 2 and predictions.shape[1] == 1:
        heart_disease_probability = predictions[0][0]  
        st.write(f"Heart Disease Risk Probability: {heart_disease_probability:.2f}")
        st.write(f"Heart Disease Risk (Binary): {'Yes' if heart_disease_probability > 0.5 else 'No'}")
    else:
        st.error("Unexpected shape of prediction output. Check the model and input data.")
