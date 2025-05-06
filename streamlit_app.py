import streamlit as st
import pickle
import numpy as np
import pandas as pd


# تحميل الموديل والسكيلر
model = pickle.load(open('svc_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def classify(prediction):
    return 'Has Heart Disease' if prediction == 1 else 'No Heart Disease'

def main():
    st.title("Heart Disease Prediction App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Prediction using SVM</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

def main():
    st.title("Heart Disease Prediction App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Prediction using SVM</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # المدخلات من المستخدم (كلها لازم تكون داخـل الدالة ومزحومة)
    age = st.slider('Age', 20, 100)
    sex = st.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    cp = st.slider('Chest Pain Type (0-3)', 0, 3)
    trtbps = st.slider('Resting Blood Pressure', 80, 200)
    chol = st.slider('Cholesterol', 100, 600)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [1, 0])
    restecg = st.slider('Resting ECG Results (0-2)', 0, 2)
    thalachh = st.slider('Max Heart Rate Achieved', 70, 210)
    exng = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [1, 0])
    oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 6.0)
    slp = st.slider('Slope of the peak exercise ST segment', 0, 2)
    caa = st.slider('Number of major vessels (0-4)', 0, 4)
    thall = st.slider('Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)', 0, 2)

# تجهيز البيانات للتنبؤ
    features = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
    thalachh, exng, oldpeak, slp, caa, thall]])
    scaled_features = scaler.transform(features)

    if st.button('Predict'):
        prediction = model.predict(scaled_features)[0]
        st.success(f'Result: {classify(prediction)}')





