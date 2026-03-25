import streamlit as st
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import numpy as np

st.set_page_config(page_title="AI Micro-Expression Pro", layout="wide")

# Initialize session state for data storage
if "data" not in st.session_state:
    st.session_state.data = []

st.title("👁️ Micro-Expression Neural Analyst")
st.write("Project by **Pallem Deva Sathwik**")

col1, col2 = st.columns([2, 1])

with col1:
    img_file = st.camera_input("Scanner Active - Capture Face")
    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        try:
            with st.spinner("Analyzing Neural Patterns..."):
                # Analyze using DeepFace
                res = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
                emo = res[0]['dominant_emotion']
                st.session_state.data.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Emotion": emo,
                    "Confidence": round(res[0]['emotion'][emo], 2)
                })
                st.success(f"Detection Successful: {emo.upper()}")
        except Exception as e:
            st.error("Face scan failed. Ensure clear lighting and try again.")

with col2:
    st.subheader("⚡ Live Metrics")
    if st.session_state.data:
        latest = st.session_state.data[-1]
        st.metric("Dominant State", latest["Emotion"].upper(), f"{latest['Confidence']}%")
        st.dataframe(pd.DataFrame(st.session_state.data).tail(5))
