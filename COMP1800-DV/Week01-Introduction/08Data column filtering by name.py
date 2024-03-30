import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())
print(data.describe())

selected = ['A', 'F', 'L']

print(data[selected].head())
print(data.head())

data[selected].plot.box()
plt.show()
data[selected].plot.density()
plt.show()
data[selected].plot.line()
plt.show()
