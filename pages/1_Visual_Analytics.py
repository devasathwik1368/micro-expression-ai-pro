import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Behavioral Analytics")

if not st.session_state.data:
    st.warning("No data found. Capture expressions on the Main Page first.")
else:
    df = pd.DataFrame(st.session_state.data)
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Emotion Intensity Timeline")
        st.line_chart(df.set_index("Timestamp")["Confidence"])
    with c2:
        st.write("### Emotion Distribution")
        fig, ax = plt.subplots()
        df['Emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
