# Imports
import re
import pandas as pd
import numpy as np
# Visualización
# import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn
# Modelos
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st

#Funciones
def clasificar_nombre(nombre):
    if ', Master.' in nombre:
        return 'kids'
    elif ', Miss.' in nombre:
        return 'miss'
    else:
        return 'adults'
    
#Carga de datos
URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(URL, index_col='PassengerId')

df = titanic.copy()

#Limpieza de datos
df.drop(columns=['Ticket', 'Cabin'], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

#Feature Engineering
df["Acompaniantes"] = df["SibSp"] + df["Parch"]
df['Acompaniantes'] = np.where(df['Acompaniantes'] > 0, 1, 0)
df.drop(columns=["SibSp", "Parch"], inplace=True)


df['Rango_edad'] = df['Name'].apply(clasificar_nombre)

mean_male = df[df['Sex'] == 'male']['Age'].mean()
mean_female = df[df['Sex'] == 'female']['Age'].mean()
mean_kids = df[df['Rango_edad'] == 'kids']['Age'].mean()
mean_miss = df[df['Rango_edad'] == 'miss']['Age'].mean()

is_male = df['Sex'] == 'male'
is_female = df['Sex'] == 'female'
is_kids = df['Rango_edad'] == 'kids'
is_miss = df['Rango_edad'] == 'miss'

is_nan = df['Age'].isna()

df.loc[is_male & is_nan, 'Age'] = mean_male
df.loc[is_female & is_nan, 'Age'] = mean_female
df.loc[is_kids & is_nan, 'Age'] = mean_kids
df.loc[is_miss & is_nan, 'Age'] = mean_miss

df.drop(columns=['Rango_edad','Name'], inplace=True)

df['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
df.rename(columns={'Sex':'is_male'}, inplace=True)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2, random_state=73)

# Escalado y dummies
X_train['Fare'] = np.log(X_train['Fare']+1)
X_test['Fare'] = np.log(X_test['Fare']+1)

esc = MinMaxScaler()
X_train[['Age', 'Fare']] = esc.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = esc.transform(X_test[['Age', 'Fare']])

map_embarke = {"S":0, "C":1, "Q":2}
X_train["Embarked"] = X_train["Embarked"].replace(map_embarke)
X_test["Embarked"] = X_test["Embarked"].replace(map_embarke)

# Cross Validation + BaseLines
modelos = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=False)
    
}

metricas = ["accuracy", "f1_macro", "recall_macro", "precision_macro", "roc_auc_ovr"]

resultados_dict = {}

for nombre_modelo, modelo in modelos.items():
    cv_resultados = cross_validate(modelo, X_train, y_train, cv=5, scoring=metricas)
    
    for metrica in metricas:
        clave = f"{nombre_modelo}_{metrica}"
        resultados_dict[clave] = cv_resultados[f"test_{metrica}"].mean()

resultados_df = pd.DataFrame([resultados_dict])

res = resultados_df.T.sort_values(by=0, ascending=False)

#Entrenar
catb = CatBoostClassifier(verbose=False)
catb.fit(X_train, y_train)

# Prediccion
y_pred = catb.predict(X_test)

# Validación del modelo
print(classification_report(y_test, y_pred))

# Optimización del modelo
param_grid = {
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [30, 50, 100]
}
catb = CatBoostClassifier(verbose=False)

grid_search = GridSearchCV(estimator=catb, param_grid=param_grid, cv=3, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_

st.title('El Titanic')

from PIL import Image

image = Image.open('titanic.jpeg')
st.image(image)

st.info("El RMS Titanic fue un transatlántico británico, el mayor barco de pasajeros del mundo al finalizar su construcción, \
    que naufragó en las aguas del océano Atlántico durante la noche del 14 y la madrugada del 15 de abril de 1912,\
        mientras realizaba su viaje inaugural desde Southampton a Nueva York, tras chocar con un iceberg. \
            En el hundimiento murieron 1496 personas de las 2208 que iban a bordo, lo que convierte a esta catástrofe en uno de los naufragios más mortales \
                de la historia ocurridos en tiempos de paz.", icon="ℹ️")

st.subheader('Vamos a intentar predecir si un pasajero sobrevivió o no')

st.subheader('Ejemplo de pasajeros del Titanic')

# Selectbox
a = st.selectbox('Selecciona el número de filas para visualizar del lote inicial',options=('5', '20', '50'))

st.write(df.head(int(a)))

st.subheader('Tras la limpieza de datos y preparación del dataset, queda el siguiente dataset de entrenamiento')

st.write(X_train)

st.subheader('Tras realizar el CV, nos aparecen los siguientes valores para los modelos y métricas usados')

st.write(res)

st.subheader('Como se puede ver, el mejor es ' + res[0])