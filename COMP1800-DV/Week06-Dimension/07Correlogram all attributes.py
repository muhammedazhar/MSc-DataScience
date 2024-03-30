import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

sns.pairplot(summary_data, height=1.5, plot_kws={'s': 20})
plt.show()
