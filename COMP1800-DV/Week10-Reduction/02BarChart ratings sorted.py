import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ratings = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/Ratings.csv', index_col=0)
print(ratings.mean(axis=1).sort_values())

data = ratings.transpose()
data = data.reindex(data.sum().sort_values().index, axis=1)

# data.sum().plot.bar(width=0.8, rot=0, figsize=(8, 8))
plt.figure(figsize=(8, 8))
x_pos = np.arange(len(data.columns))
plt.bar(x_pos, data.mean(), align='center')
plt.xticks(x_pos, data.columns)
plt.xlabel('Products', fontsize=18)
plt.ylabel('Average rating', fontsize=18)
plt.title('Average product rating', fontsize=20)
plt.show()
