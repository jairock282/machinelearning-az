import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Cargamos datos
dataset = pd.read_csv(
    "/machinelearning-az/datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values  # Utilizamos esta indexacion para obtener una matriz y no un vector ya que es la "matriz de caracteristicas"
y = dataset.iloc[:, 2].values

# Variables scale
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(np.reshape(y, (10, 1)))

# Creamos modelo de regresion
regression = SVR(kernel="rbf") # Utilizamos kernel Gaussiano
regression.fit(X, np.reshape(y, (10,)))

# Prediciton
y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform([y_pred])

#inverse scaling
X_tmp = sc_X.inverse_transform(X)
y_tmp = sc_y.inverse_transform(y)

# Visualization
X_grid = np.arange(min(X_tmp), max(X_tmp), 0.1)  #Extend point per sample
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X_tmp, y_tmp, color="red")
#plt.plot(X_grid, regression.predict(X_grid), color="blue")

plt.title("SVM")
plt.xlabel("Employee position")
plt.ylabel("Earns $")
plt.show()
