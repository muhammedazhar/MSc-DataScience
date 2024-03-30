# see  https://pandas.pydata.org/pandas-docs/version/0.13.1/visualization.html#autocorrelation-plot
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

row = 0
col = 0
fig, axes = plt.subplots(figsize=(8, 8), nrows=5, ncols=5)
fig.suptitle('Autocorrelation plots', fontsize=20, position=(0.5, 1.0))
for name in data.columns:
    sub = pd.plotting.autocorrelation_plot(data[name], axes[row, col])
    sub.set_title('Product ' + name, fontsize=10)
    sub.xaxis.label.set_visible(False)
    sub.yaxis.label.set_visible(False)
    col += 1
    if col == 5:
        row += 1
        col = 0
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
