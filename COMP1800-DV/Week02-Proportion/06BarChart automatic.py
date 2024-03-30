import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)

categories = ['High', 'Medium', 'Low', 'Very Low']
categories_selected = [[] for i in range(len(categories))]
for name in data.columns:
    total_sales = data[name].sum()
    if total_sales > 100000:
        category = 0
    elif total_sales > 40000:
        category = 1
    elif total_sales > 10000:
        category = 2
    else:
        category = 3
    categories_selected[category].append(name)
    print('Product ' + name + ' is ' + categories[category] + ' volume')

for i in range(len(categories)):
    print(f'{categories[i]}: {categories_selected[i]}')

for i, selected in enumerate(categories_selected):
    plt.figure(figsize=(8, 8))
    x_pos = np.arange(len(data[selected].columns))
    plt.bar(x_pos, data[selected].sum(), align='center')
    plt.xticks(x_pos, data[selected].columns)
    plt.xlabel('Products', fontsize=18)
    plt.ylabel('Units sold', fontsize=18)
    plt.title(categories[i] + ' Volume Product Sales', fontsize=20)
    plt.show()
