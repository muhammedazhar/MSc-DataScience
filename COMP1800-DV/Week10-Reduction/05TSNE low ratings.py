import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.mean(axis=1).sort_values())

raw_data = ratings
# raw_data = raw_data / raw_data.max()  # to normalise the data, if required
group = ['K', 'A', 'E', 'G', 'J', 'X']

raw_data_group = raw_data.loc[group]
group_size = raw_data_group.shape[0]
raw_data_other = raw_data.drop(group)
grouped_data = pd.concat([raw_data_group, raw_data_other])
print(grouped_data)

counter = 1
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Ratings - tSNE embeddings', fontsize=14, position=(0.5, 1.0))
perplexities = [6, 8, 10, 12]
for p in perplexities:
    embedding = TSNE(n_components=2, random_state=0, perplexity=p).fit_transform(grouped_data.values)
    embedded_group = embedding[:group_size, :]
    embedded_other = embedding[group_size:, :]
    sub = fig.add_subplot(2, 2, counter)
    sub.set_title(f'Perplexity = {p}', fontsize=10)
    sub.scatter(embedded_group[:, 0], embedded_group[:, 1])
    sub.scatter(embedded_other[:, 0], embedded_other[:, 1])
    # for i, name in enumerate(normalised_data.index):
    #     sub.annotate(name, (embedding[i, 0] + 5, embedding[i, 1]))
    counter += 1
fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.tight_layout()
plt.show()
