import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.mean(axis=1).sort_values())

k = 4
raw_data = ratings
# raw_data = raw_data / raw_data.max()  # to normalise the data, if required
annotate = False
perplexities = [6, 8, 10, 12]

clusters = []
k_means = KMeans(n_clusters=k, init='k-means++', random_state=1)
raw_data['label'] = k_means.fit_predict(raw_data)
for c in range(k):
    cluster = raw_data[raw_data['label'] == c]
    print(f'cluster{c} = {list(cluster.index)}')
    clusters.append(cluster.drop(['label'], axis=1))
clustered_data = pd.concat(clusters)

counter = 1
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Ratings - tSNE embeddings', position=(0.5, 1.0))
for p in perplexities:
    embedding = TSNE(n_components=2, random_state=0, perplexity=p).fit_transform(clustered_data.values)
    embedded_clusters = []
    cluster_start = 0
    for cluster in clusters:
        cluster_end = cluster_start + cluster.shape[0]
        embedded_clusters.append(embedding[cluster_start:cluster_end, :])
        cluster_start = cluster_end
    sub = fig.add_subplot(2, 2, counter)
    sub.set_title(f'Perplexity = {p}', fontsize=10)
    for embedded_cluster in embedded_clusters:
        sub.scatter(embedded_cluster[:, 0], embedded_cluster[:, 1])
    if annotate:
        for i, name in enumerate(clustered_data.index):
            sub.annotate(name, (embedding[i, 0] + 5, embedding[i, 1]))
    counter += 1
fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.tight_layout()
plt.show()
