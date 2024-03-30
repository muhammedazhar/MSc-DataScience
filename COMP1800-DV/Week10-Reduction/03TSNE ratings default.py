import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.mean(axis=1).sort_values())

raw_data = ratings
# raw_data = raw_data / raw_data.max()  # to normalise the data, if required

embedding = TSNE(n_components=2, random_state=0).fit_transform(raw_data.values)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1])
for i, name in enumerate(raw_data.index):
    plt.annotate(name, (embedding[i, 0] + 5, embedding[i, 1]))
plt.show()
