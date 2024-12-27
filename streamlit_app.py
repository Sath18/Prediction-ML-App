import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning Prediction App')

st.info('This is a ML Prediction App')

with st.expander('Data'):
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df
