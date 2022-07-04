# XGBoost (gradient boosting)
"""
Algoritmo muy eficiente para grandes conjuntos de datos
"""
# Artificial Neural Networks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Codificar datos
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score  # Crear datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier


# 1.- Data pre-processing
# Gets data
dataset = pd.read_csv("/mnt/SSD2/linux/Documents/cursos/machine_learning_A-Z/machinelearning-az/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Coding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],
    remainder='passthrough'
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]  # Remove column of dummy variables

#  Splits dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit XG-Boost
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Check test and evaluate model
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

#Aplicar k-fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std() # Que tanto variaría +- el promedio (su estabilidad en los resultados)
