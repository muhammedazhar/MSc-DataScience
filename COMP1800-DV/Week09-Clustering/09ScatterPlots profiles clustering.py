import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D scatter plot, even though not directly used

profiles = pd.read_csv('https://tinyurl.com/ChrisCoDV/CustomerProfiles.csv')
print(profiles.head())
print(profiles.describe())
print(profiles.info())

k = 6
selected = ['Age', 'Income', 'Spending']

k_means = KMeans(n_clusters=k, init='k-means++', random_state=1)
profiles['label'] = k_means.fit_predict(profiles[selected])
clusters = []
for c in range(k):
    clusters.append(profiles[profiles['label'] == c])

plt.figure(figsize=(8, 8))
for c in range(k):
    plt.scatter(clusters[c]['Age'], clusters[c]['Income'])
plt.xlabel('Age', fontsize=20)
plt.ylabel('Income', fontsize=20)
plt.show()

plt.figure(figsize=(8, 8))
for c in range(k):
    plt.scatter(clusters[c]['Age'], clusters[c]['Spending'])
plt.xlabel('Age', fontsize=20)
plt.ylabel('Spending', fontsize=20)
plt.show()

plt.figure(figsize=(8, 8))
for c in range(k):
    plt.scatter(clusters[c]['Income'], clusters[c]['Spending'])
plt.xlabel('Income', fontsize=20)
plt.ylabel('Spending', fontsize=20)
plt.show()

fig = plt.figure(figsize=(10, 6))
sub = fig.add_subplot(111, projection='3d')
for c in range(k):
    sub.scatter(clusters[c]['Age'], clusters[c]['Income'], clusters[c]['Spending'], s=60)
# sub.view_init(30, 185)
sub.set_xlabel('Age')
sub.set_ylabel('Income')
sub.set_zlabel('Spending')
plt.show()
