# adapted from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
#
# seaborn heatmap doesn't work correctly with matplotlib 3.1.1 (cuts the edges of the square off)
#  see https://github.com/mwaskom/seaborn/issues/1773
#
# need an older (or newer) version of matplotlib so close down PyCharm and run
#  pip install matplotlib==3.1.0 --force-reinstall

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)

plt.figure(figsize=(8, 8))
corr = data.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(220, 20, n=200), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
