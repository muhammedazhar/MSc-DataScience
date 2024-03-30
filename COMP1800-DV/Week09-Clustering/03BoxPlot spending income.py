import matplotlib.pyplot as plt
import pandas as pd

profiles = pd.read_csv('https://tinyurl.com/ChrisCoDV/CustomerProfiles.csv')
print(profiles.head())
print(profiles.describe())

selected = ['Age', 'Income', 'Spending']

plt.figure(figsize=(8, 8))
plt.boxplot(profiles[selected], labels=selected)
plt.xlabel('Variable', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.title('Variable distributions', fontsize=20)
plt.show()
