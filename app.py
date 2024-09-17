import pickle
import pandas as pd
import streamlit as st

def get_user_inputs():
    st.header("Prever Diabetes (TESTE)")

    age = st.number_input("Idade?", min_value=0, max_value=100)
    sex = st.selectbox("Sexo", ["Masculino", "Feminino"])
    chestpain = st.number_input("Dor no peito? (0 = Nenhuma, 1 = Controlavel, 2 = Irritante, 3 = Muito Dolorido)", min_value=0, max_value=3, step=1)
    restingblood = st.number_input("Pressão arterial em repouso?", min_value=0, max_value=200)
    fastingbloodsg = st.selectbox("O seu açúcar no sangue em jejum é maior que 120 mg/dl?", ["No", "Yes"])
    restingeletro = st.number_input("Resultados eletrocardiográficos em repouso? (valores 0,1,2)", min_value=0, max_value=2, step=1)
    heartrate = st.number_input("Batimentos cardiacos maximos atingidos?", min_value=0, max_value=202, step=1)

    sex = 0 if sex == "Female" else 1
    fastingbloodsg = 0 if fastingbloodsg == "No" else 1

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

pickle_in = open("heart.pkl", "rb")
classifier = pickle.load(pickle_in)

user_data = get_user_inputs()

if st.button("Predict"):
    predictions = classifier.predict(user_data)
    
    st.write("Predictions:", predictions)
    st.write("Predictions shape:", predictions.shape)
   
    st.write("First few prediction values:", predictions[:5] if len(predictions) > 5 else predictions)
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

