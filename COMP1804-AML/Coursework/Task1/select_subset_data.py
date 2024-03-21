# Import necessary libraries
import pandas as pd

# Define the size of the subset to select
subset_size = 100

# Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Create a subset dataframe
subset_df = df.head(subset_size)

# Write the subset dataframe to a new CSV file
subset_df.to_csv('subset_dataset.csv', index=False)

print('A subset of the data has been written to a new CSV file: subset_dataset.csv.')