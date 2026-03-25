# 👁️ AI Micro-Expression Neural Analyst

A real-time behavioral intelligence application that combines **Computer Vision** and **Large Language Models (LLMs)** to detect, track, and analyze human micro-expressions.

## 🚀 Live Demo
[Check out the Live App here!](microexpdetai.streamlit.app)

## 🛠️ Features
- **Dual-Mode Input:** Supports live webcam scanning and local image uploads.
- **Neural Emotion Detection:** Uses `DeepFace` (VGG-Face) to identify 7 universal emotions (Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral).
- **Behavioral Analytics:** Real-time data visualization of emotional intensity and sentiment distribution using `Matplotlib`.
- **Llama-3 Insights:** Integrates **Groq Cloud API** to provide expert-level behavioral reports, negotiation strategies, and stress assessments based on captured data.

## 🏗️ Technical Architecture


1. **Frontend:** Streamlit (Multi-page Web App)
2. **Computer Vision:** OpenCV & DeepFace (TensorFlow backend)
3. **AI Logic:** Llama-3-8b (via Groq API)
4. **Data Handling:** Pandas & Session State for real-time logging.

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/devasathwik1368/micro-expression-ai-pro.git](https://github.com/devasathwik1368/micro-expression-ai-pro.git)
   cd micro-expression-ai-pro
