import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv(
    "/machinelearning-az/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Utilizar dendrograma (Podemos ver que nos indica que tendríamos que crear 5 clusters)
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrograma")
plt.xlabel("Clients")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar cluster jerarquico a nuestro conjunto de datos
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

#Visualizacion de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c="red", label="Cluster 1")  #Dispercion de los puntos
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c="green", label="Cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c="cyan", label="Cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c="magenta", label="Cluster 5")
plt.title("Clusters de clientes")
plt.xlabel("Ingresos $")
plt.ylabel("Puntuacion de gasto (1-100)")
plt.legend()
plt.show()
e