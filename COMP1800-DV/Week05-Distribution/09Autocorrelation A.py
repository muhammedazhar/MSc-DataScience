# see  https://pandas.pydata.org/pandas-docs/version/0.13.1/visualization.html#autocorrelation-plot
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

pd.plotting.autocorrelation_plot(data['A'])
plt.title('Product A autocorrelation')
plt.show()
