import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
print(data.head())

selected = []
columns = data.columns
data['Others'] = [0] * len(data.index)
for name in columns:
    total_sales = data[name].sum()
    if total_sales > 10000:
        selected.append(name)
    else:
        data['Others'] += data[name]
selected.append('Others')
print(data[selected].head())

plt.figure(figsize=(8, 8))
plt.pie(data[selected].sum(), labels=selected)
plt.title('Total Product Sales', fontsize=20)
plt.legend(loc=2)
plt.show()
