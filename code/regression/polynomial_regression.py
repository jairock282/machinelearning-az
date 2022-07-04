import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures

dataset = pd.read_csv(
    "/machinelearning-az/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values  # Utilizamos esta indexacion para obtener una matriz y no un vector ya que es la "matriz de caracteristicas"
y = dataset.iloc[:, 2].values

# Adjust linear regression with the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Adjust polinomial linear regression
# Get polynomial features matrix
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
# set regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Linear regression results visualization
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Linear regression")
plt.xlabel("Employee position")
plt.ylabel("Earns $")
plt.show()

# Polynomial regression results visualization
X_grid = np.arange(min(X), max(X), 0.1)  #Extend point per sample
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Polynomial regression")
plt.xlabel("Employee position")
plt.ylabel("Earns $")
plt.show()

# Specific prediction
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
