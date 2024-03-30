import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
print(data.head())

# sort the data according to the sum of each column
data = data.reindex(data.sum().sort_values(ascending=False).index, axis=1)
print(data.head())

plt.figure(figsize=(8, 8))
x_pos = np.arange(len(data.columns))
plt.bar(x_pos, data.sum(), align='center')
plt.xticks(x_pos, data.columns)
plt.xlabel('Products', fontsize=18)
plt.ylabel('Units sold', fontsize=18)
plt.title('Total Product Sales', fontsize=20)
plt.show()
