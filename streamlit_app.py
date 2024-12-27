import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning Prediction App')

st.info('This is a ML Prediction App')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data = df, x = 'bill_length_mm', y = 'body_mass_g', color = 'species')

with st.sidebar:
  st.header('Input Features')
  island = st.selectbox(
    'Island',
    ('Biscoe','Dream','Torgersen')
  )
  bill_length_mm = st.slider(
    'Bill Length (mm)',
    32.1, 59.6, 43.9
  )
  bill_depth_mm = st.slider(
    'Bill Depth (mm)',
    13.1, 21.5, 17.2
  )
  flipper_length_mm = st.slider(
    'Flipper Length (mm)',
    172.0, 231.0, 201.0
  )
  body_mass_g = st.slider(
    'Body Mass (g)',
    2700.0, 6300.0, 4207.0
  )
  gender = st.selectbox(
    'Gender',
    ('male','female')
  )

  data = {'island' : island,
         'bill_length_mm' : bill_length_mm,
         'bill_depth_mm' : bill_depth_mm,
         'flipper_length_mm' : flipper_length_mm,
         'body_mass_g' : body_mass_g,
         'sex' : gender}

  input_df = pd.DataFrame(data, index = [0])
  input_penguins = pd.concat([input_df, X_raw], axis = 0)

encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix = encode)

target_mapper = {'Adelie': 0,
                 'Chinstrap' : 1,
                 'Gentoo' : 2}

def target_encoder(value):
  return target_mapper[value]

y = y_raw.apply(target_encoder)

input_row = df_penguins[:1]

with st.expander('Input Features'):
  st.write('**Input Features of the Penguin**')
  input_df

with st.expander('Encoded Values'):
  st.write('**Encoded Input Values**')
  input_row
  st.write('**Encoded y**')
  y

classifier = RandomForestClassifier()
classifier.fit(X_raw, y)

prediction = classifier.predict(input_row)
prediction_probability = classifier.predict_proba(input_row)

prediction_probability
