import streamlit as st
import numpy as np
import pickle
import pandas as pd

# تحميل الموديل والسكيلر
model = pickle.load(open('svc_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# دالة التصنيف
def classify(prediction):
    return 'Has Heart Disease' if prediction == 1 else 'No Heart Disease'

# الواجهة الرئيسية
def main():
    st.title("Heart Disease Prediction App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Prediction using SVM</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # الحقول المدخلة من المستخدم (نفس 11 اللي تدرب عليهم الموديل)
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
    thall = st.slider('Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)', 0, 2)

    # التجهيز للتنبؤ
    features = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                          thalachh, exng, oldpeak, thall]])
    scaled_features = scaler.transform(features)

    if st.button('Predict'):
        prediction = model.predict(scaled_features)[0]
        st.success(f'Result: {classify(prediction)}')

if __name__ == '__main__':
    main()
