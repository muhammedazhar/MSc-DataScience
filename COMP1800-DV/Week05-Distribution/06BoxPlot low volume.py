import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['D', 'E', 'M', 'O', 'P', 'T', 'X']

plt.figure(figsize=(8, 8))
# data[selected].boxplot()
plt.boxplot(data[selected].transpose(), labels=selected)
plt.xlabel('Product', fontsize=18)
plt.ylabel('Units sold per day', fontsize=18)
plt.title('Low volume product sales distributions', fontsize=20)
plt.show()
