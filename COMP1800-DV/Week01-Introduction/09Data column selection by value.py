import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = data.columns[data.sum() > 100000]

data = data[selected]
print(data.head())

data.plot.box()
plt.show()
data.plot.density()
plt.show()
data.plot.line()
plt.show()
