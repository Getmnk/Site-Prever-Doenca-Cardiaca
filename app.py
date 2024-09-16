import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image

pickle_in = open("heart.pkl","rb")
classifier=pickle.load(pickle_in)

def get_user_inputs():
    st.header("Heart Disease Prediction Input (TEST, take it with a grain of salt)")
    
    age = st.number_input("Age?", min_value=0, max_value=100)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chestpain = st.number_input("Chest pain? (0 = none, 1 = bearable, 2 = unbearable, 3 = very painful)", min_value=0, max_value=3, step=1)
    restingblood = st.number_input("Resting blood pressure?", min_value=0, max_value=200)
    fastingbloodsg = st.selectbox("Is your fasting blood sugar bigger than 120 mg/dl?", ["No", "Yes"])
    restingeletro = st.number_input("Resting electrocardiographic results (values 0,1,2)", min_value=0, max_value=2, step=1)
    heartrate = st.number_input("Maximum heart rate achieved?", min_value=0, max_value=202, step=1)
    
    sex = 0 if sex == "Female" else 1 
    fastingbloodsg == 0 if fastingbloodsg == "No" else 1
    
    data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "chestpain": [chestpain],
        "restingblood": [restingblood],
        "fastingbloodsg": [fastingbloodsg],
        "restingeletro": [restingeletro],
        "heartrate": [heartrate]
    })

    return data
    
user_data = get_user_inputs()

if st.button("Predict"):
    predictions = pickle.predict(user_data)
    
    st.write("Predictions shape:", predictions.shape)
    st.write("Predictions data:", predictions)

    if len(predictions.shape) == 2 and predictions.shape[1] == 1:
        heart_disease_probability = predictions[0][0]  
        st.write(f"Heart Disease Risk Probability: {heart_disease_probability:.2f}")
        st.write(f"Heart Disease Risk (Binary): {'Yes' if heart_disease_probability > 0.5 else 'No'}")
    else:
        st.error("Unexpected shape of prediction output. Check the model and input data.")
