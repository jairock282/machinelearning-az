e
from sklearn.cluster import KMeans


dataset = pd.read_csv("/mnt/SSD2/linux/Documents/cursos/machine_learning_A-Z/machinelearning-az/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

#Metodo del codo para definir el numero de clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar el metodo Kmeans con el numero optimo de clusters
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizacion de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 1")  #Dispercion de los puntos
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="cyan", label="Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="magenta", label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="Baricentros")
plt.title("Clusters de clientes")
plt.xlabel("Ingresos $")
plt.ylabel("Puntuacion de gasto (1-100)")
plt.legend()
plt.show()



