import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load the training dataset
df_train = pd.read_csv('train_dataset.csv')

# Separating features and labels
df_train.dropna(inplace=True)
X_train = df_train.drop('category', axis=1)
y_train = df_train['category']

# Convert to numpy array for model training
X_train_values = X_train.values
y_train_values = y_train.values

# Train the Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_values, y_train_values)

# Save the trained model
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb, f)

# Load the testing dataset
df_test = pd.read_csv('test_dataset.csv')

# Ensure the same correction is made for the testing dataset
df_test.dropna(inplace=True)
X_test = df_test.drop('category', axis=1)
y_test = df_test['category']

# Convert to numpy arrays for model training
X_test_values = X_test.values
y_test_values = y_test.values

# Make predictions on the testing dataset
y_pred = nb.predict(X_test_values)

# Evaluate model performance using various metrics
print('Classification report:\n', classification_report(y_test_values, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test_values, y_pred))
print('Accuracy score:', accuracy_score(y_test, y_pred))