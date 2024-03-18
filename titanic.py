# Tratamiento de datos
import re
import pandas as pd
import numpy as np
# Visualización
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# Modelos
# from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, \
# roc_curve, roc_auc_score, ConfusionMatrixDisplay, multilabel_confusion_matrix
import streamlit as st
import os

print(os.getcwd())
st.title('El Titanic')

from PIL import Image

image = Image.open('titanic.jpeg')
st.image(image)

st.info("El RMS Titanic fue un transatlántico británico, el mayor barco de pasajeros del mundo al finalizar su construcción, que naufragó en las aguas del océano Atlántico durante la noche del 14 y la madrugada del 15 de abril de 1912, mientras realizaba su viaje inaugural desde Southampton a Nueva York, tras chocar con un iceberg. En el hundimiento murieron 1496 personas de las 2208 que iban a bordo, lo que convierte a esta catástrofe en uno de los naufragios más mortales de la historia ocurridos en tiempos de paz.", icon="i")

st.subheader('Ejemplo de pasajeros del Titanic')
URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(URL, index_col='PassengerId')

df = titanic.copy()
st.write(df.head())