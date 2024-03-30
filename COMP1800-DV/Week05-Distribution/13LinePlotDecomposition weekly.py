import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)

selected = ['H', 'M', 'O']

for name in selected:
    result = seasonal_decompose(data[name], model='multiplicative', period=7)
    result.plot()
    plt.suptitle('Product ' + name, position=(0.5, 1.0))
    plt.show()
