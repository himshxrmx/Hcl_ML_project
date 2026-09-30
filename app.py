import joblib
import pandas as pd
import streamlit as st

model = joblib.load('models/resume_model.pkl')
scaler = joblib.load('models/scaler.pkl')
metadata = joblib.load('models/metadata.pkl')

st.title('AI Resume Screening')
st.caption(f"Model: {metadata['model_name']}")

years_experience = st.number_input('Years of experience', 0, 40, 5)
skills_match_score = st.slider('Skills match score', 0.0, 100.0, 75.0)
education_level = st.selectbox('Education level', list(metadata['edu_mapping'].keys()))
project_count = st.number_input('Project count', 0, 50, 10)
resume_length = st.number_input('Resume length', 50, 2000, 550)
github_activity = st.number_input('GitHub activity', 0, 2000, 300)

if st.button('Screen candidate'):
    candidate = pd.DataFrame({
        'years_experience': [years_experience],
        'skills_match_score': [skills_match_score],
        'education_level': [metadata['edu_mapping'][education_level]],
        'project_count': [project_count],
        'resume_length': [resume_length],
        'github_activity': [github_activity]
    })[metadata['feature_order']]

    prob = model.predict_proba(scaler.transform(candidate))[0, 1]

    if prob >= 0.5:
        st.success(f"Shortlisted ({prob*100:.1f}% probability)")
    else:
        st.error(f"Not shortlisted ({prob*100:.1f}% probability)")
