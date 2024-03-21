# Import necessary libraries
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Convert text to lowercase
df['paragraph'] = df['paragraph'].str.lower()

# Remove punctuation
df['paragraph'] = df['paragraph'].str.replace('[^\w\s]','')

# Tokenization, stop words removal
stop = stopwords.words('english')
df['paragraph'] = df['paragraph'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['paragraph']).toarray()
Y = df['category']

# Split the dataset into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Save the datasets
df_train = pd.DataFrame(X_train)
df_train['category'] = Y_train
df_train.to_csv('train_dataset.csv', index=False)

df_test = pd.DataFrame(X_test)
df_test['category'] = Y_test
df_test.to_csv('test_dataset.csv', index=False)