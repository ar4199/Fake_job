import streamlit as st
import joblib 
import shap
import numpy as np


pipe = joblib.load('fake_job_detector.pkl')
vectorizer = pipe.named_steps['tfidf']
model = pipe.named_steps['clf']

@st.cache_resource
def get_explainer(_X_sample):
    return shap.Explainer(model,X_sample)

st.title('Fake job Detector')

job_text= st.text_area('Paste Job Description Here', height = 300)

if st.button('Analyze'):
    if not job_text.strip():
        st.warining('Please Paste A Job Description')
    else:
        proba = pipe.predict_proba([job_text])[0][1]
        pred_label ='Fake' if proba >= 0.5 else 'Real'
        st.markdown(f'Prediction: {pred_label}')
        st.markdown(f'Probability of being Fake:`{proba*100:.2f}%`')


        X_sample =vectorizer.transform([job_text])
        explainer=get_explainer(vectorizer.transform([job_text]))
        shap_values = explainer(X_sample)

        nonzero_indices = X_sample.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        shap_vals =shap_values[0].values[nonzero_indices]
        words =feature_names[nonzero_indices]
        word_contribs= sorted(zip(words, shap_vals), key=lambda x: abs(x[1]), reverse = True)

        st.markdown('Top Keywords Influencing the Prediction:')
        for word, val in word_contribs[:10]:
            direction = '🔺' if val>0 else '🔻' 
            st.markdown(f'{direction}`{word}`->SHAP:`{val:.4f}`')