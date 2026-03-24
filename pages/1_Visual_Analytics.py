import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Behavioral Analytics Dashboard")

if not st.session_state.data:
    st.warning("No data found. Please use the camera on the Main Page first.")
else:
    df = pd.DataFrame(st.session_state.data)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Emotion Timeline")
        st.line_chart(df.set_index("Timestamp")["Score"])
        
    with c2:
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots(facecolor='#0E1117')
        df['Emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, textprops={'color':"w"})
        st.pyplot(fig)

    st.divider()
    st.download_button("Download Report Data", df.to_csv(), "micro_expression_data.csv")
