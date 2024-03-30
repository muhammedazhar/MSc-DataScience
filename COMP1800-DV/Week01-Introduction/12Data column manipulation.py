import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['A', 'F', 'L']

data = data[selected]
print(data.head())

data['A + F'] = data['A'] + data['F']
data['L + 500'] = data['L'] + 500
data['L * 2'] = data['L'] * 2
data = data.drop(selected, axis=1)
print(data.head())

data.plot.line()
plt.show()

