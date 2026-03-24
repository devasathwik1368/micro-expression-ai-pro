import streamlit as st
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import numpy as np

st.set_page_config(page_title="AI Micro-Expression Pro", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00FFAA; }
    </style>
    """, unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = []

st.title("👁️ Micro-Expression Neural Analyst")
st.write("Student Project by Pallem Deva Sathwik")

col_left, col_right = st.columns([2, 1])

with col_left:
    img_file = st.camera_input("Scan Face for Analysis")
    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        try:
            with st.spinner("Analyzing expressions..."):
                res = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
                emo = res[0]['dominant_emotion']
                st.session_state.data.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Emotion": emo,
                    "Score": round(res[0]['emotion'][emo], 2)
                })
                st.success(f"Detected Emotion: {emo.upper()}")
        except Exception as e:
            st.error("Could not process image. Please ensure your face is visible.")

with col_right:
    st.subheader("Live Session Metrics")
    if st.session_state.data:
        st.metric("Current State", st.session_state.data[-1]["Emotion"].upper())
        st.dataframe(pd.DataFrame(st.session_state.data).tail(5))
    else:
        st.info("Awaiting camera input...")
