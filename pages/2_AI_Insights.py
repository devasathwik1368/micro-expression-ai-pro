import streamlit as st
import pandas as pd
from groq import Groq

st.title("🤖 AI Behavioral Insight")

if not st.session_state.data:
    st.error("Missing data. Please run the live feed first.")
else:
    # Handle API Key via Secrets (Recommended) or Input
    api_key = st.secrets.get("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key", type="password")

    if st.button("Generate Professional Report"):
        if not api_key:
            st.warning("Please provide a Groq API Key.")
        else:
            client = Groq(api_key=api_key)
            df = pd.DataFrame(st.session_state.data)
            counts = df['Emotion'].value_counts().to_dict()
            
            prompt = f"""
            Analyze these facial micro-expression statistics: {counts}
            Provide a 3-part professional assessment:
            1. Negotiation Insights: (Likelihood of agreement, signs of hesitation).
            2. Mental Health Assessment: (Current stress level and emotional stability).
            3. Tactical Advice: (How should a negotiator or therapist proceed?).
            """
            
            with st.spinner("Llama-3 is analyzing behavioral patterns..."):
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                )
                st.markdown(chat_completion.choices[0].message.content)
