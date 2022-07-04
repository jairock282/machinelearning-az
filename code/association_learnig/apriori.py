import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("/code/association_learnig")
from apyori import apriori

dataset = pd.read_csv("/mnt/SSD2/linux/Documents/cursos/machine_learning_A-Z/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/Market_Basket_Optimisation.csv", header=None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Entrenar algoritmo apriori
rules = apriori(transactions,
                min_support=0.003,  # Con que precensia minima debe aparecer un item para ser consideradp
                min_confidence=0.2, # muy permisivo = reglas muy obvias | muy estricto = no se encuentran reglas
                min_lift=3, # Cociente entre el valor de support y confidence
                min_length=2)  #Cantidad minima de productos para ser considerada asociación


# visualization
results = list(rules)
print(results[3])
