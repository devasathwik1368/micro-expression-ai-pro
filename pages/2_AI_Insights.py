import streamlit as st
import pandas as pd
from groq import Groq

st.title("🤖 Llama-3 Behavioral Insights")

if not st.session_state.data:
    st.error("No facial data found. Run the scanner on the home page first.")
else:
    # Uses Streamlit Secrets for security
    if "GROQ_API_KEY" in st.secrets:
        if st.button("Generate Expert AI Report"):
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            counts = pd.DataFrame(st.session_state.data)['Emotion'].value_counts().to_dict()
            
            prompt = f"Analyze these detected micro-expressions: {counts}. Provide negotiation advice and stress assessment."
            
            with st.spinner("AI is analyzing behavioral patterns..."):
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                )
                st.markdown("### AI Assessment Report")
                st.write(chat_completion.choices[0].message.content)
    else:
        st.error("GROQ_API_KEY not found in Streamlit Secrets!")
