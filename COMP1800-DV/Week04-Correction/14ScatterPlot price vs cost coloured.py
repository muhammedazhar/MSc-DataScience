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

summary_data_high = summary_data.loc[data.sum() > 100000]
summary_data_medium = summary_data.loc[(data.sum() > 40000) & (data.sum() <= 100000)]
summary_data_low = summary_data.loc[data.sum() <= 40000]

plt.figure(figsize=(8, 8))
plt.scatter(summary_data_high['Price'], summary_data_high['Cost'])
plt.scatter(summary_data_medium['Price'], summary_data_medium['Cost'])
plt.scatter(summary_data_low['Price'], summary_data_low['Cost'])
plt.ylim(ymin=0)
plt.title('Price per unit vs Cost per unit', fontsize=20)
plt.xlabel('Price (£ per unit)', fontsize=18)
plt.ylabel('Cost (£ per unit)', fontsize=18)
plt.legend(['High', 'Medium', 'Low'], loc=2)
plt.show()
