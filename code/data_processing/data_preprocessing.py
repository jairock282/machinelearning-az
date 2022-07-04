import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # Eliminar nans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Codificar datos
from sklearn.model_selection import train_test_split # Crear datasets
from sklearn.preprocessing import StandardScaler

# Platilla pre procesado

# importar dataset
dataset = pd.read_csv("./machinelearning-az/datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Tratamiento de NAs
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Codificar datos categoricos
labelencoder_X = LabelEncoder()  # Codificador de datos
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X), dtype=float)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Dividir el data set en conjunto de entraneminto y conjunto de testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Escalado de variables
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

