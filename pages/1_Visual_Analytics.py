import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Behavioral Analytics Dashboard")

if not st.session_state.data:
    st.warning("No data found. Capture expressions on the Main Page first.")
else:
    df = pd.DataFrame(st.session_state.data)
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Emotion Timeline")
        st.line_chart(df.set_index("Timestamp")["Intensity"])
    with c2:
        st.write("### Distribution")
        fig, ax = plt.subplots(facecolor='#0E1117')
        df['Emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, textprops={'color':"w"})
        st.pyplot(fig)
