import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

plt.figure(figsize=(8, 8))
# data.plot.area(figsize=(8, 8))
plt.stackplot(data.index, data.transpose())
plt.xlabel('Date', fontsize=18)
plt.ylabel('Units sold', fontsize=18)
plt.title('Stacked Product Sales', fontsize=20)
plt.legend(data.columns, loc=2)
plt.show()
