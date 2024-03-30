import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.head())
print(ratings.mean(axis=1))

k = 4
selected = ratings.columns

k_means = KMeans(n_clusters=k, init='k-means++', random_state=1)
ratings['label'] = k_means.fit_predict(ratings[selected])
clusters = []
for c in range(k):
    clusters.append(ratings[ratings['label'] == c])

plt.figure(figsize=(8, 8))
for c in range(k):
    plt.scatter(clusters[c]['Rating 1'], clusters[c]['Rating 2'], s=100)
for i, name in enumerate(ratings.index):
    plt.annotate(name, (ratings['Rating 1'][i] + 0.05, ratings['Rating 2'][i]))
plt.xlabel('Rating 1', fontsize=20)
plt.ylabel('Rating 2', fontsize=20)
plt.show()

plt.figure(figsize=(8, 8))
for c in range(k):
    plt.scatter(clusters[c]['Rating 3'], clusters[c]['Rating 8'], s=100)
for i, name in enumerate(ratings.index):
    plt.annotate(name, (ratings['Rating 3'][i] + 0.05, ratings['Rating 8'][i]))
plt.xlabel('Rating 3', fontsize=20)
plt.ylabel('Rating 8', fontsize=20)
plt.show()
