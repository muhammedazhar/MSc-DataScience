import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D scatter plot, even though not directly used

profiles = pd.read_csv('https://tinyurl.com/ChrisCoDV/CustomerProfiles.csv')
print(profiles.head())
print(profiles.describe())

fig = plt.figure(figsize=(10, 6))
sub = fig.add_subplot(111, projection='3d')
sub.scatter(profiles['Age'], profiles['Income'], profiles['Spending'], s=60)
# sub.view_init(30, 5)
sub.set_xlabel('Age')
sub.set_ylabel('Income')
sub.set_zlabel('Spending')
plt.show()
