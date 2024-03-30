import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)

selected = ['D', 'E', 'M', 'O', 'P', 'T', 'X']
print(data[selected].head())

# data[selected].sum().plot.bar(width=0.8, rot=0, figsize=(8, 8))
plt.figure(figsize=(8, 8))
x_pos = np.arange(len(data[selected].columns))
plt.bar(x_pos, data[selected].sum(), align='center')
plt.xticks(x_pos, data[selected].columns)
plt.xlabel('Products', fontsize=18)
plt.ylabel('Units sold', fontsize=18)
plt.title('Low Volume Product Sales', fontsize=20)
plt.show()
