# Instalar pkts
#pip install Orange3
#pip install PyQt5
# Importar librerías
import pickle
import Orange
#import pandas as pd
import numpy as np
# import plotly.express as px
# import PIL
import streamlit as st

#Cargar modelo .pkcls
model = pickle.load(open('/content/Logistic_IRIS.pkcls', 'rb'))
#Guardar y cargar modelo como .pkl
pickle.dump(model, open('/content/Logistic_IRIS.pkl', 'wb'))
model_loaded=pickle.load(open('/content/Logistic_IRIS.pkl', 'rb'))

# Interfaz de entrada de datos con streamlit
st.set_page_config(page_title='ML Model')
st.header('Modelo ML')
st.subheader('Características de entrada')

sepal_length = st.number_input('Insert a sepal length')
sepal_width = st.number_input('Insert a sepal width')
petal_length = st.number_input('Insert a petal length')
petal_width = st.number_input('Insert a petal width')
#data={'sepal length':sepal_length, 'sepal width':sepal_width,
#      'petal length':petal_length, 'petal width':petal_width}
#data_=pd.DataFrame(data)
data=[sepal_length, sepal_width, petal_length, petal_width]

#data = Orange.data.Table("/content/iris_val_4.xlsx")
#Pronóstico de modelo
#Pred= model(data)

# https://www.youtube.com/watch?v=vibDbEBnyV4
x_in = np.asarray(data).reshape(1, -1)
pred = model_loaded.predict(x_in)
species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
species_names = [species_dict[int(pred[0])]]
#df=pd.DataFrame(species_names)
#print(species_names[0])
#print(df[0])

#Salida página web en Streamlit
st.subheader('Pronóstico Iris')
st.text(species_names[0])
#st.dataframe(df)
