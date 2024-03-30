import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())
print(data.sum(axis=1))

x_min = 2000
x_max = 3500
bin_width = 50

n_bins = int((x_max - x_min) / bin_width)
print(f'{n_bins} bins')
bins = [(x_min + x * bin_width) for x in range(n_bins + 1)]

plt.figure(figsize=(8, 8))
plt.hist(data.sum(axis=1), bins, edgecolor='w')
plt.title('Daily total sales distribution', fontsize=20)
plt.show()
