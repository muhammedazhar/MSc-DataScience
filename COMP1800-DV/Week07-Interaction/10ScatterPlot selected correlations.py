import pandas as pd
import hvplot.pandas

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

xlimits = (20, 230)
ylimits = (20, 120)
plot = data.hvplot.scatter(
    frame_height=300, frame_width=300,
    x='H', y='M', xlabel='', ylabel='',
    xlim=xlimits, ylim=ylimits, size=10
) * \
data.hvplot.scatter(
    frame_height=300, frame_width=300,
    x='H', y='O', xlabel='', ylabel='',
    xlim=xlimits, ylim=ylimits, size=10
) * \
data.hvplot.scatter(
    frame_height=300, frame_width=300,
    x='M', y='O', xlabel='', ylabel='',
    xlim=xlimits, ylim=ylimits, size=10
)
hvplot.show(plot)
