import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning Prediction App')

st.info('This is a ML Prediction App')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**y**')
  y = df.species
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data = df, x = 'bill_length_mm', y = 'body_mass_g', color = 'species')

with st.sidebar:
  st.header('Input Features')
  island = st.selectbox(
    'Island',
    ('Biscoe','Dream','Torgersen')
  )
  gender = st.selectbox(
    'Gender',
    ('Male','Female')
  )
  bill_length_mm = st.slider(
    'Bill Length (mm)',
    32.1, 59.6, 43.9
  )
