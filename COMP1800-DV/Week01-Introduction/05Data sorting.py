import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)

# sort the data according to the maximum value of each row, smallest first
data = data.reindex(data.max(axis=1).sort_values().index)
print(data.max(axis=1))
print(data.head())

# sort the data according to the sum of each column, largest first
data = data.reindex(data.sum().sort_values(ascending=False).index, axis=1)
print(data.sum())
print(data.head())
