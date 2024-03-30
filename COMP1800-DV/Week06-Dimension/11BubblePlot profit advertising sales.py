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
summary_data['Advertising'] = summary_data['Marketing'] / summary_data['Sales']
summary_data['Cost'] = summary_data['Price'] - summary_data['Profit']
print(summary_data.head())
print(summary_data.describe())

summary_data['BubbleSize'] = summary_data['Sales'] * 0.01

plt.figure(figsize=(8, 8))
plt.scatter(summary_data['Profit'], summary_data['Advertising'], s=summary_data['BubbleSize'], alpha=0.5)
plt.title('Profit vs Advertising (vs Sales)', fontsize=20)
plt.xlabel('Profit per unit (£)', fontsize=18)
plt.ylabel('Advertising per unit (£)', fontsize=18)
for i, name in enumerate(summary_data.index):
    plt.annotate(name, (summary_data['Profit'][i] + 0.1, summary_data['Advertising'][i]))
plt.plot([0, 25], [2, 2], linestyle=':', color='black', label='£2.00 per unit')
plt.plot([0, 25], [1, 1], linestyle=':', color='red', label='£1.00 per unit')
plt.plot([0, 25], [0.5, 0.5], linestyle=':', color='orange', label='£0.50 per unit')
plt.plot([0, 25], [0.25, 0.25], linestyle=':', color='green', label='£0.25 per unit')
plt.plot([0, 15], [0, 15], color='magenta', label='profit = advertising (break even)')
plt.legend(loc=2, title='Marketing spend limits')
plt.show()
