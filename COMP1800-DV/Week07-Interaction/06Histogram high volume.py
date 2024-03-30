import pandas as pd
import hvplot.pandas

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print(data.head())

selected = ['A', 'F', 'L']

plot = data[selected].hvplot.hist(
    frame_height=500, frame_width=500,
    xlabel='Units sold per day', ylabel='Frequency',
    title='High Volume Products',
    alpha=0.5, muted_alpha=0, muted_fill_alpha=0, muted_line_alpha=0,
    tools=['pan', 'box_zoom', 'wheel_zoom', 'undo', 'redo', 'hover', 'save', 'reset']
)
hvplot.show(plot)
