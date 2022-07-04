# Artificial Neural Networks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Codificar datos
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  # Crear datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense


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

# Scales values (Must in ANN)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# 2.- Create the ANN
# Init ANN
classifier = Sequential()

# NN architecture
# Input layer and first hidden layer
classifier.add(
    Dense(units=6,
          kernel_initializer="uniform",
          activation="relu",
          input_dim=11)  # Input nodes
)
# second hidden layer
classifier.add(
    Dense(units=6,
          kernel_initializer="uniform",
          activation="relu")
)
# output layer
classifier.add(
    Dense(units=1,
          kernel_initializer="uniform",
          activation="sigmoid")
)

# compile ANN
classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# fit ANN
classifier.fit(
    X_train, y_train,
    batch_size=10,
    epochs=100
)

# Check test and evaluate model
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)

