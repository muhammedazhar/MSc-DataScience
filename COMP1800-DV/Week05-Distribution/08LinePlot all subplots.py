import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
pd.plotting.register_matplotlib_converters()
data.index = pd.to_datetime(data.index)
print(data.head())

counter = 1
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Product sales', fontsize=20, position=(0.5, 1.0))
for name in data.columns:
    sub = fig.add_subplot(5, 5, counter)
    sub.set_title('Product ' + name, fontsize=10)
    sub.plot(data.index, data[name], linewidth=0.5)
    sub.axes.get_xaxis().set_ticks([])  # remove the x ticks
    counter += 1
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
