import streamlit as slt
import numpy as np
import pickle


slt.title('Heart Disease Prediction Machine Learning Model')
user_input = slt.text_area("Enter the heart data configurations")
if slt.button('Predict if you have heart disease or not'):
    with open('LRM.pkl', 'rb') as file:
        model = pickle.load(file)
    input_2d = np.array([user_input.split(',')],dtype=float)
    prediction = model.predict(input_2d)
    if prediction[0] == 0:
        slt.header("You are healthy!!😄")
    elif prediction[0] == 1:
        slt.header("Please visit a cardiologist😟")