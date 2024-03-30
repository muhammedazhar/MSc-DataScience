import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)

counter = 1
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Product sales distributions', fontsize=20, position=(0.5, 1.0))
for name in data:
    sub = fig.add_subplot(5, 5, counter)
    sub.set_title('Product ' + name, fontsize=10)
    sub.hist(data[name], edgecolor='w')
    counter += 1
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
