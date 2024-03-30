import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)

sample_rate = 'W'  # weekly
averaged_data = data.resample(sample_rate).mean()

selected = ['A', 'F', 'L']
print(data[selected].head())
print(averaged_data[selected].head())

plt.figure(figsize=(8, 8))
plt.plot(data[selected], linewidth=0.5)
plt.gca().set_prop_cycle(None)
plt.plot(averaged_data[selected], linewidth=2)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Units sold', fontsize=18)
plt.title('High Volume Sales with Weekly Averages', fontsize=20)
plt.legend(selected, loc=2)
plt.show()
