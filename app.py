import pickle
import pandas as pd
import streamlit as st
import numpy as py

def get_user_inputs():
    st.header("Prever Doença Cardiaca (TESTE)")

    age = st.number_input("Idade?", min_value=0, max_value=100)
    sex = st.selectbox("Sexo", ["Masculino", "Feminino"])
    chestpain = st.number_input("Dor no peito? (1 = Nenhuma, 2 = Controlavel, 3 = Irritante, 4 = Muito Dolorido)", min_value=1, max_value=4, step=1)
    restingblood = st.number_input("Pressão arterial em repouso?", min_value=0, max_value=200, step=1)
    chol = st.number_input("Colesterol?", min_value=126, max_value=564, step=1)
    fastingbloodsg = st.selectbox("O seu açúcar no sangue em jejum é maior que 120 mg/dl?", ["Não", "Sim"])
    restingeletro = st.number_input("Resultados eletrocardiográficos em repouso? (valores 0,1,2)", min_value=0, max_value=2, step=1)
    heartrate = st.number_input("Batimentos cardiacos maximos atingidos?", min_value=0, max_value=202, step=1)
    angina = st.selectbox("Você tem dor toracica?", ["Não", "Sim"])
    stdepression = st.number_input("Depressão do segmento ST induzida pelo exercicio em relação ao repouso?", min_value=0.0, max_value=6.2, step=0.1)
    stslope = st.number_input("Inclinação do ST?", min_value=1, max_value=3, step=1)
    
    sex = 0 if sex == "Feminino" else 1
    fastingbloodsg = 0 if fastingbloodsg == "Não" else 1
    angina = 0 if angina == "Não" else 1
    
    data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "chestpain": [chestpain],
        "restingblood": [restingblood],
        "chol": [chol],
        "fastingbloodsg": [fastingbloodsg],
        "restingeletro": [restingeletro],
        "heartrate": [heartrate],
        "angina": [angina],
        "stdepression": [stdepression],
        "stslope": [stslope],
    })

    return data

with open("heart.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

user_data = get_user_inputs()

if st.button("Prever"):
    predictions = classifier.predict(user_data)
    
    st.write("Previsões:", predictions)
    st.write("Formato das previsões:", predictions.shape)
   
    st.write("Primeiros valores das previsões:", predictions[:5] if len(predictions) > 5 else predictions)
    
    if len(predictions.shape) == 2:
        if predictions.shape[1] == 1:
            heart_disease_probability = predictions[0][0]
            st.write(f"Probabilidade de Doença Cardiaca: {heart_disease_probability:.2f}")
            st.write(f"Doença Cardiaca (Binario): {'Sim' if heart_disease_probability > 0.5 else 'Não'}")
        else:
            st.error("ERRO! Tamanho inesperado da segunda dimensão nas previsões.")
    elif len(predictions.shape) == 1:
        st.write(f"Doença Cardiaca (Binario): {'Sim' if predictions[0] == 1 else 'Não'}")
    else:
        st.error("ERRO NO MODELO!")
