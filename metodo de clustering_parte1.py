import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Definir la data
data = {
    'Estado': ['AL', 'AZ', 'AR', 'CA', 'CT', 'DE', 'DC', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'ND', 'OH', 'OK', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
    'Paquetes': [4.96213, 4.66312, 5.10709, 4.50449, 4.66983, 5.04705, 4.65637, 4.80081, 4.97974, 4.74902, 4.81445, 5.11129, 4.80857, 4.79263, 5.37906, 4.98602, 4.98722, 4.77751, 4.73877, 4.94744, 4.69589, 4.9399, 5.0643, 4.73313, 4.77558, 4.96642, 5.1099, 4.70633, 4.58107, 4.66496, 4.58237, 4.97952, 4.7272, 4.80363, 4.84693, 5.07801, 4.81545, 5.04939, 4.65398, 4.40859, 5.08799, 4.93065, 4.66134, 4.82454, 4.83026, 5.00087],
    'Precio': [0.20487, 0.1664, 0.23406, 0.36399, 0.32149, 0.21929, 0.28946, 0.28733, 0.12826, 0.17541, 0.24806, 0.08992, 0.24081, 0.21642, -0.0326, 0.23856, 0.29106, 0.12575, 0.22613, 0.23067, 0.34297, 0.13638, 0.08731, 0.15303, 0.18907, 0.32304, 0.15852, 0.30901, 0.16458, 0.34701, 0.18197, 0.12889, 0.19554, 0.22784, 0.30324, 0.07944, 0.13139, 0.15547, 0.28196, 0.1926, 0.18018, 0.11818, 0.35053, 0.12008, 0.22954, 0.10029],
    'Ingreso': [4.64039, 4.68389, 4.59435, 4.88147, 5.09472, 4.87087, 5.0596, 4.81155, 4.73299, 4.64307, 4.90387, 4.72916, 4.74211, 4.79613, 4.64937, 4.61461, 4.75501, 4.94692, 4.99998, 4.8062, 4.81207, 4.52938, 4.78189, 4.70417, 4.79671, 4.83816, 5.00319, 5.10268, 4.58202, 4.96075, 4.69163, 4.75875, 4.6273, 4.83516, 4.8467, 4.62549, 4.67747, 4.72525, 4.73437, 4.55586, 4.77578, 4.8549, 4.85645, 4.56859, 4.75826, 4.71169]
}

# Convertir la data en un DataFrame de pandas
datos = pd.DataFrame(data)

# Seleccionar las variables relevantes para el clustering
datos_cluster = datos.iloc[:, 1:]  # Excluir la columna "Estado"

# Determinar el número óptimo de clusters
wcss = []  # Lista para almacenar la suma de los cuadrados dentro de los clusters
for i in range(1, 11):  # Probamos con un rango de 1 a 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)  # Inicializamos el modelo KMeans
    kmeans.fit(datos_cluster)  # Ajustamos el modelo a los datos
    wcss.append(kmeans.inertia_)  # Añadimos la suma de los cuadrados dentro de los clusters a la lista

# Gráfico del método del codo
plt.plot(range(1, 11), wcss, marker='o')  # Graficamos el número de clusters vs. la suma de los cuadrados dentro de los clusters
plt.title('Método del codo')  # Título del gráfico
plt.xlabel('Número de clusters')  # Etiqueta del eje x
plt.ylabel('WCSS')  # Etiqueta del eje y
plt.xticks(range(1, 11))  # Definimos los ticks del eje x
plt.grid(True)  # Activamos la cuadrícula en el gráfico
plt.show()  # Mostramos el gráfico

# Realizar el clustering con el número óptimo de clusters
numero_clusters = 3  # Seleccionamos el número óptimo de clusters basado en el análisis anterior
kmeans = KMeans(n_clusters=numero_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)  # Inicializamos el modelo KMeans con el número óptimo de clusters
pred_y = kmeans.fit_predict(datos_cluster)  # Ajustamos el modelo y predecimos los clusters para cada muestra

# Visualizar los resultados del clustering
pca = PCA(n_components=2)  # Inicializamos PCA para reducir la dimensionalidad de los datos a 2 dimensiones
principalComponents = pca.fit_transform(datos_cluster)  # Aplicamos PCA a los datos
principalDf = pd.DataFrame(data=principalComponents, columns=['componente_principal_1', 'componente_principal_2'])  # Creamos un DataFrame con las componentes principales

finalDf = pd.concat([principalDf, pd.DataFrame({'cluster':pred_y})], axis=1)  # Concatenamos las componentes principales con los clusters predichos

# Identificación de grupos por colores
colores = ['tab:red', 'tab:green', 'tab:blue']  # Definimos colores para los clusters

# Imprimir los resultados
print("Resultados del clustering:")
print(finalDf)

# Gráfico de dispersión con colores por grupo
plt.figure(figsize=(10, 6))  # Definimos el tamaño del gráfico
for i in range(numero_clusters):  # Iteramos sobre cada cluster
    plt.scatter(finalDf[finalDf['cluster']==i]['componente_principal_1'], 
                finalDf[finalDf['cluster']==i]['componente_principal_2'], 
                c=colores[i], label=f'Cluster {i+1}', s=50)  # Graficamos los puntos de cada cluster con un color distinto

# Anotar los nombres de los estados en el gráfico
for i, txt in enumerate(datos['Estado']):  # Iteramos sobre cada estado
    plt.annotate(txt, (finalDf['componente_principal_1'][i], finalDf['componente_principal_2'][i]), fontsize=8)  # Anotamos el nombre de cada estado en el gráfico

plt.title('Clustering de Estados según Consumo de Cigarrillos')  # Título del gráfico
plt.xlabel('Componente Principal 1')  # Etiqueta del eje x
plt.ylabel('Componente Principal 2')  # Etiqueta del eje y
plt.legend(loc='best')  # Mostrar la leyenda en la mejor posición
plt.grid(True)  # Activar la cuadrícula en el gráfico
plt.show()  # Mostrar el gráfico
