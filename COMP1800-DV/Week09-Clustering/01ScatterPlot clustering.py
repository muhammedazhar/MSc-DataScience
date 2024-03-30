import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
print(data.head())

marketing_data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/MarketingPerProduct.csv', index_col=0)
price_per_unit = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/PricePerUnit.csv', index_col=0)
profit_per_unit = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/ProfitPerUnit.csv', index_col=0)

summary_data = pd.DataFrame(index=data.columns)
summary_data['Price'] = price_per_unit.values
summary_data['Profit'] = profit_per_unit.values
summary_data['Sales'] = data.sum().values
summary_data['Marketing'] = marketing_data.values
summary_data['Cost'] = summary_data['Price'] - summary_data['Profit']
print(summary_data.head())
print(summary_data.describe())

k = 4
selected = ['Marketing', 'Sales']
k_means = KMeans(n_clusters=k, init='k-means++', random_state=1)
summary_data['label'] = k_means.fit_predict(summary_data[selected])

clusters = []
for c in range(k):
    clusters.append(summary_data[summary_data['label'] == c])

plt.figure(figsize=(8, 8))
for c in range(k):
    plt.scatter(clusters[c]['Marketing'], clusters[c]['Sales'], s=100)
for i, name in enumerate(summary_data.index):
    plt.annotate(name, (summary_data['Marketing'][i] + 500, summary_data['Sales'][i]))
plt.xlabel('Marketing')
plt.ylabel('Sales')
plt.show()
