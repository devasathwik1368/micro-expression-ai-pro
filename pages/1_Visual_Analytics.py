import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Behavioral Analytics")

if not st.session_state.data:
    st.warning("No session data detected. Please capture expressions on the home page.")
else:
    df = pd.DataFrame(st.session_state.data)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("### Emotion Timeline")
        st.line_chart(df, x="Timestamp", y="Intensity")
        
    with col_b:
        st.write("### Intensity Distribution")
        fig, ax = plt.subplots(facecolor='#0E1117')
        df['Emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, textprops={'color':"w"})
        st.pyplot(fig)

    st.divider()
    st.download_button("Download Session CSV", df.to_csv(), "micro_expression_report.csv")
