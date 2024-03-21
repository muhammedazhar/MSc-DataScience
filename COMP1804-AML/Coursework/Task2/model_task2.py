import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('subset_dataset.csv')

# Encode the 'text_clarity' column
df['text_clarity'] = df['text_clarity'].apply(lambda x: 0 if x == 'clear_enough' else 1)

# Separate the features and target variable
X = df['paragraph']
y = df['text_clarity']

# Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical vectors
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create the model
svm = SVC(kernel='linear', probability=True, random_state=42)

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)

# Evaluate the model's performance
print('Classification report:\n', classification_report(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Accuracy score:', accuracy_score(y_test, y_pred))