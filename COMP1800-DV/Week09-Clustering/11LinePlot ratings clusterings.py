import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.head())
print(ratings.mean(axis=1))

selected = ratings.columns
min_k = 2
max_k = 10
k_values = range(min_k, max_k + 1)
WCSS = []
silhouette = []
for k in k_values:
    k_means = KMeans(n_clusters=k, init='k-means++', random_state=1)
    labels = k_means.fit_predict(ratings[selected])
    WCSS.append(k_means.inertia_)
    silhouette.append(silhouette_score(ratings[selected], labels))
WCSS = np.asarray(WCSS) / max(WCSS)

plt.figure(figsize=(12, 6))
plt.plot(k_values, WCSS, marker='o', label='within cluster sum of squares')
plt.plot(k_values, silhouette, marker='X', label='average silhouette score')
plt.xlabel('number of clusters (k)')
plt.legend()
plt.grid()
plt.show()
