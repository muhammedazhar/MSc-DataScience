import pandas as pd
import hvplot.pandas
import holoviews as hv

data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/DailySales.csv', index_col=0)
print(data.head())

marketing_data = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/MarketingPerProduct.csv', index_col=0)
price_per_unit = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/PricePerUnit.csv', index_col=0)
profit_per_unit = pd.read_csv('https://tinyurl.com/ChrisCoDV/Products/ProfitPerUnit.csv', index_col=0)

summary_data = pd.DataFrame(index=data.columns)
summary_data['Price'] = price_per_unit.values
summary_data['Profit'] = profit_per_unit.values
summary_data['Sales'] = data.sum().values
summary_data['Marketing'] = marketing_data.values
summary_data['Advertising'] = summary_data['Marketing'] / summary_data['Sales']
summary_data['Cost'] = summary_data['Price'] - summary_data['Profit']
print(summary_data.head())
print(summary_data.describe())

summary_data['BubbleSize'] = summary_data['Sales'] * 0.01

hline200 = hv.HLine(2.00, label='£2.00 per unit').opts(line_dash='dashed', line_width=0.5, color='black')
hline100 = hv.HLine(1.00, label='£1.00 per unit').opts(line_dash='dashed', line_width=0.5, color='red')
hline050 = hv.HLine(0.50, label='£0.50 per unit').opts(line_dash='dashed', line_width=0.5, color='orange')
hline025 = hv.HLine(0.25, label='£0.25 per unit').opts(line_dash='dashed', line_width=0.5, color='green')

plot = summary_data.hvplot.scatter(
    frame_height=500, frame_width=500,
    title='Profit vs Advertising (vs Sales)',
    xlabel='Profit (£/unit)', ylabel='Advertising (£/unit)',
    alpha=0.5, padding=0.1, hover_cols='all',
    tools=['pan', 'box_zoom', 'wheel_zoom', 'undo', 'redo', 'hover', 'save', 'reset'],
    x='Profit', y='Advertising', size='BubbleSize'
) * \
hline200 * \
hline100 * \
hline050 * \
hline025
hvplot.show(plot)
