import matplotlib.pyplot as plt
import seaborn as sns
sns(style="ticks")

# Load the example dataset for Anscombe's quartet
data = sns.load_dataset("anscombe")
# print(data)
# print(data[data['dataset'] == 'I'].describe())
# print(data[data['dataset'] == 'II'].describe())
# print(data[data['dataset'] == 'III'].describe())
# print(data[data['dataset'] == 'IV'].describe())

# Show the results of a linear regression within each dataset
sns.lmplot(x='x', y='y', col='dataset', hue="dataset", data=data,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1}, fit_reg=False)
plt.show()