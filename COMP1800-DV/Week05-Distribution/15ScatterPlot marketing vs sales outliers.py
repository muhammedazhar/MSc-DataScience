import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
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

plt.figure(figsize=(8, 8))
plt.scatter(summary_data['Marketing'], summary_data['Sales'])
plt.title('Marketing spend vs Sales', fontsize=20)
plt.xlabel('Marketing spend (£)', fontsize=18)
plt.ylabel('Sales', fontsize=18)
for i, name in enumerate(data.columns):
    plt.annotate(name, (summary_data['Marketing'][i] + 300, summary_data['Sales'][i]))
plt.plot([0, 40000], [0, 20000], linestyle=':', color='black', label='£2.00 per unit')
plt.plot([0, 40000], [0, 40000], linestyle=':', color='red', label='£1.00 per unit')
plt.plot([0, 40000], [0, 80000], linestyle=':', color='orange', label='£0.50 per unit')
plt.plot([0, 40000], [0, 160000], linestyle=':', color='green', label='£0.25 per unit')
plt.legend(loc=2, title='Marketing spend limits')
plt.show()
