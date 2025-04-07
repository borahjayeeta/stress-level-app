
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Level Predictor")
st.write("Enter the values below to predict stress level:")

features = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]

user_input = {feature: st.slider(feature.replace('_', ' ').title(), 0, 10, 5) for feature in features}

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Stress Level: {prediction:.2f}")
