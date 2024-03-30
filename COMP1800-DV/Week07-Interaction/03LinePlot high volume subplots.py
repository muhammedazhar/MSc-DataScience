import pandas as pd
import hvplot.pandas

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['A', 'F', 'L']

plot = data[selected].hvplot.line(
    frame_height=300, frame_width=300,
    xlabel='Date', ylabel='Units sold',
    title='High Volume Products',
    subplots=True
)
hvplot.show(plot)
