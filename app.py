import streamlit as st
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Micro-Expression Pro", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM UI CUSTOMIZATION ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00FFAA; font-family: monospace; }
    .stButton>button { width: 100%; border-radius: 20px; border: 1px solid #00FFAA; background-color: transparent; color: white; }
    .stButton>button:hover { background-color: #00FFAA; color: black; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if "data" not in st.session_state:
    st.session_state.data = []

st.title("👁️ Micro-Expression Neural Analyst")
st.caption("Advanced Computer Vision & Llama-3 Behavioral Intelligence")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📡 Live Feed")
    img_file = st.camera_input("Scanner Active")
    
    if img_file:
        with st.spinner("Processing Micro-movements..."):
            file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            try:
                # DeepFace analysis (CPU Optimized)
                res = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
                emo = res[0]['dominant_emotion']
                score = res[0]['emotion'][emo]
                
                # Store entry
                st.session_state.data.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Emotion": emo,
                    "Intensity": round(score, 2)
                })
                st.success(f"Detection Successful: {emo.upper()}")
            except Exception as e:
                st.error(f"Analysis Error: {e}")

with col_right:
    st.subheader("⚡ Real-time Metrics")
    if st.session_state.data:
        latest = st.session_state.data[-1]
        st.metric("Dominant State", latest["Emotion"].upper(), f"{latest['Intensity']}% Match")
        
        st.markdown("### 📋 Session Logs")
        st.dataframe(pd.DataFrame(st.session_state.data).tail(10), use_container_width=True)
    else:
        st.info("Awaiting input scanner...")
