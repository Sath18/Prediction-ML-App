import streamlit as st

st.title('🤖 Machine Learning Prediction App')

st.info('This is a ML Prediction App')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")

df
