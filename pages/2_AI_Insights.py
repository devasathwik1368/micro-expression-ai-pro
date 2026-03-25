import streamlit as st
import pandas as pd
from groq import Groq

st.title("🤖 Llama-3 Behavioral Insights")

if not st.session_state.data:
    st.error("Capture facial data on the Home page first.")
else:
    # Safely retrieving the key from Streamlit Cloud Secrets dashboard
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        
        if st.button("Generate AI Assessment"):
            client = Groq(api_key=api_key)
            df = pd.DataFrame(st.session_state.data)
            counts = df['Emotion'].value_counts().to_dict()
            
            prompt = f"Analyze these micro-expressions: {counts}. Provide negotiation strategies and stress levels."
            
            with st.spinner("AI is thinking..."):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-8b-8192",
                    )
                    st.markdown("### 📝 AI Assessment Report")
                    st.write(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.error("API Key not found! Add GROQ_API_KEY to your Streamlit Secrets.")
