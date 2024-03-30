import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['B', 'C', 'I', 'K', 'N', 'Q', 'R', 'U', 'V', 'Y']

plt.figure(figsize=(8, 8))
# data[selected].boxplot()
plt.boxplot(data[selected], labels=selected)
plt.xlabel('Product', fontsize=18)
plt.ylabel('Units sold per day', fontsize=18)
plt.title('Very low volume product sales distributions', fontsize=20)
plt.show()
