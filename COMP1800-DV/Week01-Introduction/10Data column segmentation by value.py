import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = data.columns[data.sum() > 100000]

data_selected1 = data[selected]
data = data.drop(selected, axis=1)
print(data_selected1.head())
print(data.head())

data_selected1.plot.line()
plt.show()

selected = data.columns[data.sum() > 40000]

data_selected2 = data[selected]
data = data.drop(selected, axis=1)
print(data_selected2.head())
print(data.head())

data_selected2.plot.line()
plt.show()

selected = data.columns[data.sum() > 10000]

data_selected3 = data[selected]
data = data.drop(selected, axis=1)
print(data_selected3.head())
print(data.head())

data_selected3.plot.line()
plt.show()

data.plot.line()
plt.show()
