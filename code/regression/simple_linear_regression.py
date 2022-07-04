"""
Regresion lineal simple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importar dataset
dataset = pd.read_csv(
    "/machinelearning-az/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Create train and test batch
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# Create simple linear regression
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicts on batch test
y_pred = regression.predict(X_test)

# Training result visualization
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")  # Recta de regresion
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo en $")
plt.show()

# Test result visualization
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")  # Recta de regresion
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo en $")
plt.show()






