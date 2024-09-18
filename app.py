import pickle
import pandas as pd
import streamlit as st
import numpy as py

pickle_in = open("heart.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Bem-vindo ao Previsor de Doença Cardíaca (TESTE)"

def predict_disease(age, sex, chestpain, restingblood, chol, fastingbloodsg, restingeletro, heartrate, angina, stdepression, stslope):
    """Função para prever a doença cardíaca com base nas entradas do usuário.
    
    Args:
        age (int): Idade do usuário.
        sex (int): Sexo do usuário (0 = Feminino, 1 = Masculino).
        chestpain (int): Tipo de dor no peito.
        restingblood (int): Pressão arterial em repouso.
        chol (int): Nível de colesterol.
        fastingbloodsg (int): Açúcar no sangue em jejum.
        restingeletro (int): Resultados eletrocardiográficos em repouso.
        heartrate (int): Batimentos cardíacos máximos.
        angina (int): Indica dor torácica.
        stdepression (float): Depressão do segmento ST.
        stslope (int): Inclinação do ST.
        
    Returns:
        prediction: Resultado da previsão.
    """
    prediction = classifier.predict([[age, sex, chestpain, restingblood, chol, fastingbloodsg, restingeletro, heartrate, angina, stdepression, stslope]])
    return prediction

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Aplicativo de Previsão de Doença Cardíaca</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.number_input("Idade?", min_value=0, max_value=100)
    sex = st.selectbox("Sexo", ["Masculino", "Feminino"])
    chestpain = st.number_input("Dor no peito? (1 = Nenhuma, 2 = Controlável, 3 = Irritante, 4 = Muito Dolorido)", min_value=1, max_value=4, step=1)
    restingblood = st.number_input("Pressão arterial em repouso?", min_value=0, max_value=200, step=1)
    chol = st.number_input("Colesterol?", min_value=126, max_value=564, step=1)
    fastingbloodsg = st.selectbox("Açúcar no sangue em jejum maior que 120 mg/dl?", ["Não", "Sim"])
    restingeletro = st.number_input("Resultados eletrocardiográficos em repouso? (valores 0,1,2)", min_value=0, max_value=2, step=1)
    heartrate = st.number_input("Batimentos cardíacos máximos atingidos?", min_value=0, max_value=202, step=1)
    angina = st.selectbox("Você tem dor torácica?", ["Não", "Sim"])
    stdepression = st.number_input("Depressão do segmento ST induzida pelo exercício?", min_value=0.0, max_value=6.2, step=0.1)
    stslope = st.number_input("Inclinação do ST?", min_value=1, max_value=3, step=1)

    sex = 0 if sex == "Feminino" else 1
    fastingbloodsg = 0 if fastingbloodsg == "Não" else 1
    angina = 0 if angina == "Não" else 1

    if st.button("Prever"):
        result = predict_disease(age, sex, chestpain, restingblood, chol, fastingbloodsg, restingeletro, heartrate, angina, stdepression, stslope)
        st.success('Resultado da previsão: {}'.format("Sim" if result[0] == 1 else "Não"))

   
if __name__ == '__main__':
    main()

