import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['A', 'F', 'L']
data = data[selected]

data = data.loc[pd.to_datetime('2019-01-01'): pd.to_datetime('2019-01-07')]
print(data.head())

data.sum().plot.bar()
plt.show()

data = data.transpose()
print(data)

ax = data.sum().plot.bar()
plt.gcf().autofmt_xdate()
plt.show()
