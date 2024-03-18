# Tratamiento de datos
import re
import pandas as pd
import numpy as np
# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
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

# image = Image.open('Teoria/img/titanic.jpeg')
# st.image(image)

st.subheader('Ejemplo de pasajeros del Titanic')
URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(URL, index_col='PassengerId')

df = titanic.copy()
st.write(df.head())