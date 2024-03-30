import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.mean(axis=1).sort_values())

raw_data = ratings
# raw_data = raw_data / raw_data.max()  # to normalise the data, if required

counter = 1
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Ratings - tSNE embeddings',  position=(0.5, 1.0))
perplexities = [4, 6, 8, 10, 12, 15, 20, 25, 30]
for p in perplexities:
    embedding = TSNE(n_components=2, random_state=0, perplexity=p).fit_transform(raw_data.values)
    sub = fig.add_subplot(3, 3, counter)
    sub.set_title(f'Perplexity = {p}', fontsize=10)
    plt.scatter(embedding[:, 0], embedding[:, 1])
    # for i, name in enumerate(normalised_data.index):
    #     plt.annotate(name, (embedding[i, 0] + 5, embedding[i, 1]))
    counter += 1
fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.tight_layout()
plt.show()
