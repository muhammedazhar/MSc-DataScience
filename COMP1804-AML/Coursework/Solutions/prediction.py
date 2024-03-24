import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Load the trained models
with open('naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Preprocess the 'paragraph' field
vectorizer = TfidfVectorizer()
paragraphs = vectorizer.fit_transform(df['paragraph'])

# Predict the 'category' labels using the Naive Bayes model
nb_predictions = nb_model.predict(paragraphs)

# Predict the 'text_clarity' labels using the SVM model
svm_predictions = svm_model.predict(paragraphs)

# Save predictions to the dataframe
df['category'] = nb_predictions

# Encode the predicted text clarity labels back to the original format
df['text_clarity'] = svm_predictions.apply(lambda x: 'clear_enough' if x == 0 else 'not_clear_enough')

# Export the dataframe to CSV
df.to_csv('cleaned_dataset_with_predictions.csv', index=False)