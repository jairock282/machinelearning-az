import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Multiple Linear Regression
dataset = pd.read_csv(
    "/machinelearning-az/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Code State variable
# String to numeric data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# numeric data to code format
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)

# Remove one category from the dummy variables in order to avoid the co-dependence
X = X[:, 1:]

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
regression = LinearRegression()
regression.fit(X_train, y_train)

# Prediction
y_pred = regression.predict(X_test)

# Implement Backward elimination (MANUAL VERSION)
# the statsmodels library needs this modification
# add a column of 1s in order to set the value of b0 (y = b0 + b1*x1 + b2*x2 + ...)
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
SL = 0.05

# select the independent variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # Backwards elimination takes as first step all the variables
regression_OSL = sm.OLS(endog=y, exog=X_opt).fit()  # create our model
regression_OSL.summary()

# select the independent variables
X_opt = X[:, [0, 1, 3, 4, 5]]  # Backwards elimination takes as first step all the variables
regression_OSL = sm.OLS(endog=y, exog=X_opt).fit()  # create our model
regression_OSL.summary()

# select the independent variables
X_opt = X[:, [0, 3, 4, 5]]  # Backwards elimination takes as first step all the variables
regression_OSL = sm.OLS(endog=y, exog=X_opt).fit()  # create our model
regression_OSL.summary()

# select the independent variables
X_opt = X[:, [0, 3, 5]]  # Backwards elimination takes as first step all the variables
regression_OSL = sm.OLS(endog=y, exog=X_opt).fit()  # create our model
regression_OSL.summary()

# select the independent variables
X_opt = X[:, [0, 3]]  # Backwards elimination takes as first step all the variables
regression_OSL = sm.OLS(endog=y, exog=X_opt).fit()  # create our model
regression_OSL.summary()
# Results, the most significant variable is the R&D Spend attribute



# Implement Backward elimination (AUTOMATIC VERSION)










