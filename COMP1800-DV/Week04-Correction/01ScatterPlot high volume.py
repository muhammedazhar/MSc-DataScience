import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['A', 'F', 'L']

for i, name_i in enumerate(selected):
    for j in range(i + 1, len(selected)):
        name_j = selected[j]
        plt.figure(figsize=(8, 8))
        plt.scatter(data[name_i], data[name_j])
        plt.title('Product ' + name_i + ' vs Product ' + name_j, fontsize=20)
        plt.xlabel('Product ' + name_i, fontsize=18)
        plt.ylabel('Product ' + name_j, fontsize=18)
        plt.show()
