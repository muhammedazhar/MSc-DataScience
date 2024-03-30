import matplotlib.pyplot as plt
import pandas as pd

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

summary_data['BubbleSize'] = summary_data['Profit'] * 20

plt.figure(figsize=(8, 8))
plt.scatter(summary_data['Price'], summary_data['Cost'], s=summary_data['BubbleSize'], alpha=0.5)
plt.xticks([10, 20, 30, 40, 50])
plt.yticks([10, 20, 30, 40, 50])
plt.title('Price vs Cost (vs Profit)', fontsize=20)
plt.xlabel('Price', fontsize=18)
plt.ylabel('Cost', fontsize=18)
for i, name in enumerate(summary_data.index):
    plt.annotate(name, (summary_data['Price'][i], summary_data['Cost'][i]))
plt.plot([0, 50], [0, 50], linestyle=':', color='r', label='price = cost (zero profit)')
plt.legend(loc=2)
plt.show()
