import streamlit as st
import pandas as pd
from groq import Groq

st.title("🤖 Llama-3 Behavioral Insights")

if "data" not in st.session_state or not st.session_state.data:
    st.error("No facial data found. Run the scanner on the home page first.")
else:
    if "GROQ_API_KEY" in st.secrets:
        if st.button("Generate Expert AI Report"):
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                # CLEAN THE DATA: Convert counts to simple strings
                df = pd.DataFrame(st.session_state.data)
                counts = df['Emotion'].value_counts().to_dict()
                
                # Format the prompt clearly
                prompt = f"""
                Act as a Behavioral Expert. Analyze these detected micro-expression counts:
                {counts}
                
                Provide:
                1. Negotiation readiness score (1-10).
                2. Analysis of the subject's stress levels.
                3. Three tactical communication tips based on this emotional data.
                """
                
                with st.spinner("AI is analyzing behavioral patterns..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-8b-8192",
                        # Adding temperature to ensure a stable response
                        temperature=0.7, 
                    )
                    st.markdown("### 📝 AI Assessment Report")
                    st.write(chat_completion.choices[0].message.content)
            
            except Exception as e:
                # This will show the actual error message if it fails again
                st.error(f"Groq API Error: {str(e)}")
    else:
        st.error("GROQ_API_KEY not found in Streamlit Secrets!")
