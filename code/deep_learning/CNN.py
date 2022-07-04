import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Codificar datos
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  # Crear datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# 1.- Build CNN
# Set CNN
classifier = Sequential()

# Add convolution layer
classifier.add(
    Conv2D(
        filters=32, # Cantidad de filtros a aplicar
        kernel_size=(3, 3), #tamaño del kernel, reduce la imagen -1renglon -1columna
        input_shape=(64, 64, 3),  # Imagen a color
        activation="relu" #Funcion de activacion para eliminar la linealidad
    )
)

# Max pooling
classifier.add(MaxPool2D(pool_size=(2, 2))) #Max pooling par = reducir la imagen a la mitad

# Add second convolution layer
classifier.add(
    Conv2D(
        filters=32, # Cantidad de filtros a aplicar
        kernel_size=(3, 3), #tamaño del kernel, reduce la imagen -1renglon -1columna
        activation="relu" #Funcion de activacion para eliminar la linealidad
    )
)

# second Max pooling
classifier.add(MaxPool2D(pool_size=(2, 2))) #Max pooling par = reducir la imagen a la mitad

# Flattening
classifier.add(Flatten())

# Fully connected layer
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Compile CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 2.- Fit CNN
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory(
    '/mnt/SSD2/linux/Documents/cursos/machine_learning_A-Z/machinelearning-az/datasets/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

testing_dataset = test_datagen.flow_from_directory(
    '/mnt/SSD2/linux/Documents/cursos/machine_learning_A-Z/machinelearning-az/datasets/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

classifier.fit(
    training_dataset,
    steps_per_epoch=200,
    epochs=25,
    validation_data=testing_dataset,
    validation_steps=100
)


