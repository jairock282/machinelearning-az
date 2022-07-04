import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

# import dataset
dataset = pd.read_csv(
    "/machinelearning-az/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

#Fit model
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X.reshape(X.shape[0], 1), y)

#Prediction
y_pred = regression.predict([[6.5]])

#Visualization
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color="red")
plt.plot(X, regression.predict(X.reshape(X.shape[0], 1)), color="blue")
plt.title("Decission Tree Regression")
plt.xlabel("Employee position")
plt.ylabel("Earns ('$')")
plt.show()

