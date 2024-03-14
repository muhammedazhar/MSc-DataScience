import os
print(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle

# Assuming necessary NLTK downloads are done
file_path = 'C:/Users/muhammedazhar/Developer/MSc-DataScience/COMP1804-AML/Coursework/input/dataset.csv'
print(os.path.exists(file_path))
df = pd.read_csv(file_path, usecols=['par_id', 'paragraph', 'has_entity', 'category'])
df = df.dropna()
df['category'] = df['category'].str.lower()
df = df[df['has_entity'] != 'data missing']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_paragraph'] = df['paragraph'].apply(clean_text)
df['tokens'] = df['cleaned_paragraph'].apply(nltk.word_tokenize)

# Splitting the dataset
X = df['cleaned_paragraph']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Encoding labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Building the pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words="english")),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=.01)),
])

# Fitting the model
text_clf.fit(X_train, y_train_encoded)

# Prediction on new data
docs_new = ['He went on to win the Royal Medal of the Royal Society in 1971 and the Copley Medal in 1979.']
predicted = text_clf.predict(docs_new)
predicted_category = encoder.inverse_transform(predicted)

print(predicted_category)

# Serialization
with open('model.pkl', 'wb') as f:
    pickle.dump(text_clf, f)

# Deserialization
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

# Predict with the loaded model
predicted = clf2.predict(docs_new)
print(encoder.inverse_transform(predicted))