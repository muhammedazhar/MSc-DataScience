import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/DS_data.csv')

print(data.head())

print(data.describe())

data.plot.line()
plt.show()
data.plot.scatter('x', 'y')
plt.show()