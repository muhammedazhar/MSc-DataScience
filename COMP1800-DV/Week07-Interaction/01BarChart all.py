# this file is not used in the lecture
# but is included here as an example of an interactive bar chart
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hvplot.pandas

data = pd.read_csv('../Data/Products/DailySales.csv', index_col=0)

print(data.head())
print(data.sum())

plot = data.sum().hvplot.bar(
    frame_height=200, frame_width=600,
    xlabel='Product', ylabel='Units sold',
    title='All Products',
)
hvplot.show(plot)
