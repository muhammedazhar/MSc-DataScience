import pandas as pd
import random

# Open and read the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Select a random subset of 100 samples
df_subset = df.sample(n=100, replace=False)

# Add in 'clear_enough' and 'not_clear_enough' labels on the 'text_clarity' column based on the lexicon_count, difficult_words, and last_editor_gender
def label_clarity(row):
    if row['difficult_words'] <= row['lexicon_count']*0.03 and row['last_editor_gender'] != 'unknown':
        return 'clear_enough'
    else:
        return 'not_clear_enough'
df_subset['text_clarity'] = df_subset.apply(lambda row: label_clarity(row), axis=1)

# Write the sample data back to a CSV file
df_subset.to_csv('subset_dataset.csv', index=False)