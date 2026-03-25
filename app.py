import streamlit as st
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Micro-Expression Pro", layout="wide")

if "data" not in st.session_state:
    st.session_state.data = []

st.title("👁️ Micro-Expression Neural Analyst")
st.write("Project by **Pallem Deva Sathwik**")

# Create tabs for Live Camera vs Storage Upload
tab1, tab2 = st.tabs(["📸 Live Scanner", "📁 Upload from Storage"])

processed_image = None

with tab1:
    img_file = st.camera_input("Scanner Active")
    if img_file:
        processed_image = img_file

with tab2:
    uploaded_file = st.file_uploader("Choose an image from your device", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Selected Image", width=400)
        processed_image = uploaded_file

# Logic to process whichever image is provided
if processed_image:
    file_bytes = np.frombuffer(processed_image.getvalue(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if st.button("Run Neural Analysis"):
        try:
            with st.spinner("Analyzing Facial Patterns..."):
                res = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
                emo = res[0]['dominant_emotion']
                st.session_state.data.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Source": "Upload" if tab2 else "Camera",
                    "Emotion": emo,
                    "Confidence": round(res[0]['emotion'][emo], 2)
                })
                st.success(f"Analysis Complete: {emo.upper()}")
        except Exception as e:
            st.error("Analysis failed. Ensure the face is clearly visible.")

# Dashboard section
if st.session_state.data:
    st.divider()
    st.subheader("⚡ session Logs")
    st.dataframe(pd.DataFrame(st.session_state.data).tail(10), use_container_width=True)
