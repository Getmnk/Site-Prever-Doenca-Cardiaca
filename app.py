import pickle
import pandas as pd
import streamlit as st
import numpy as py

pickle_in = open("heart.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Bem-vindo ao Previsor de Doença Cardíaca (TESTE)"

def predict_disease(age, sex, chestpain, restingblood, chol, fastingbloodsg, restingeletro, heartrate, angina):
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
        
    Returns:
        prediction: Resultado da previsão.
    """
    prediction = classifier.predict([[age, sex, chestpain, restingblood, chol, fastingbloodsg, restingeletro, heartrate, angina]])
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
    chestpain = st.number_input("Dor no peito? (0 = Angina típica, 1 = Angina atipica, 2 = Dor não anginosa, 3 = Assintomatico)", min_value=0, max_value=3, step=1)
    restingblood = st.number_input("Pressão arterial em repouso?", min_value=0, max_value=200, step=1)
    chol = st.number_input("Nivel de colesterol sérico?", min_value=126, max_value=564, step=1)
    fastingbloodsg = st.selectbox("Açúcar no sangue em jejum maior que 120 mg/dl?", ["Não", "Sim"])
    restingeletro = st.number_input("Resultados eletrocardiográficos em repouso? 0 = Normal, 1 = anormalidade da onda ST-T (inversões da onda T e/ou elevação ou depressão do segmento ST > 0,05 mV) , 2 = mostrando hipertrofia ventricular esquerda provável ou definitiva pelos critérios de Estes", min_value=0, max_value=2, step=1)
    heartrate = st.number_input("Batimentos cardíacos máximos atingidos?", min_value=0, max_value=202, step=1)
    angina = st.selectbox("Você tem dor torácica?", ["Não", "Sim"])
    
    sex = 0 if sex == "Feminino" else 1
    fastingbloodsg = 0 if fastingbloodsg == "Não" else 1
    angina = 0 if angina == "Não" else 1

    if st.button("Prever"):
        result = predict_disease(age, sex, chestpain, restingblood, chol, fastingbloodsg, restingeletro, heartrate, angina)
        st.success('Resultado da previsão: {}'.format("Sim" if result[0] == 0.90 else "Não"))

   
if __name__ == '__main__':
    main()

