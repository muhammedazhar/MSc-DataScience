import pandas as pd
import hvplot.pandas

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['A', 'F', 'L']

plot = data[selected].hvplot.line(
    frame_height=500, frame_width=500,
    xlabel='Date', ylabel='Units sold',
    title='High Volume Products'
)
hvplot.show(plot)
