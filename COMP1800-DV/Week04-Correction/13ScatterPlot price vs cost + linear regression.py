import matplotlib.pyplot as plt
import numpy as np
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

plt.figure(figsize=(8, 8))
plt.scatter(summary_data['Price'], summary_data['Cost'])
z = np.polyfit(summary_data['Price'], summary_data['Cost'], 1)
trend = np.poly1d(z)
plt.plot(summary_data['Cost'], trend(summary_data['Cost']))
plt.title('Price per unit vs Cost per unit', fontsize=20)
plt.xlabel('Price (£ per unit)', fontsize=18)
plt.ylabel('Cost (£ per unit)', fontsize=18)
plt.show()
