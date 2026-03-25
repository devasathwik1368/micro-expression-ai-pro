import streamlit as st
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Micro-Expression AI Pro", layout="wide")

# Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00FFAA; }
    </style>
    """, unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = []

st.title("👁️ Micro-Expression Neural Analyst")
st.write("Project by Pallem Deva Sathwik")

col1, col2 = st.columns([2, 1])

with col1:
    img_file = st.camera_input("Scanner Active")
    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        try:
            with st.spinner("Analyzing Neural Patterns..."):
                res = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
                emo = res[0]['dominant_emotion']
                st.session_state.data.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Emotion": emo,
                    "Intensity": round(res[0]['emotion'][emo], 2)
                })
                st.success(f"Detection Successful: {emo.upper()}")
        except Exception as e:
            st.error("Face not detected. Please ensure clear lighting.")

with col2:
    st.subheader("⚡ Real-time Metrics")
    if st.session_state.data:
        latest = st.session_state.data[-1]
        st.metric("Current State", latest["Emotion"].upper(), f"{latest['Intensity']}%")
        st.dataframe(pd.DataFrame(st.session_state.data).tail(5))

