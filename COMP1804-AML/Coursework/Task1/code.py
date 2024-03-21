import pandas as pd
import numpy as np
import nltk
import seaborn as sns

# Optional: Setup for better visuals
sns.set_theme(style="whitegrid")

# Function to download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet'] # For tokenization, stopwords and lemmatization
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print('Resources downloaded successfully.')
        except LookupError:
            nltk.download(resource)
            print('Resources already downloaded.')
# This code is copied from my previous college project. It is called Lazy Loading NLTK Resources.

# Call the function to download resources if not already present
download_nltk_resources()

# Load the dataset for Task 1
def load_t1_df(filename):
    df = pd.read_csv(filename, usecols=['par_id', 'paragraph', 'has_entity', 'lexicon_count', 'difficult_words', 'last_editor_gender', 'category'])
    size = df.shape
    if not df.empty:
        print(f"{size} rows and columns (without `text_clarity`) loaded successfully for taks 1.")
    else:
        print("The dataset is empty.")
    return df

# Load the dataset for Task 2
def load_t2_df(filename):
    df = pd.read_csv(filename, usecols=['par_id', 'paragraph', 'has_entity', 'lexicon_count', 'difficult_words', 'last_editor_gender', 'category', 'text_clarity'])
    size = df.shape
    if not df.empty:
        print(f"{size} rows and columns loaded successfully for task 2.")
    else:
        print("The dataset is empty.")
    return df

'''
	- Function usage: df = load_t1_df('filename_with_path')
	- Replace `filename_with_path` with your original value.
'''

df = load_t1_df('dataset.csv')

def clean_df(df):
    print(f'Initial shape: {df.shape}. Checking for missing values...')
    print(df.isnull().sum(), '\n')

    # df = df.dropna()
    df.dropna(inplace=True)
    # df.dropna(subset=['category', 'difficult_words'], inplace=True)

    print(f'After removing missing values, shape: {df.shape}. Verifying no missing values remain...')
    print(df.isnull().sum())

    return df

'''
	- Function usage: df = clean_df(df)
	- Replace `df` with your original DataFrame.
'''

df = clean_df(df)

def process_df(df):
    # Combine operations for 'category' column
    print('Checking for unique values in the category column')
    print(df['category'].unique())

    # Convert the 'category' column to lowercase and print unique values again
    df['category'] = df['category'].str.lower()
    print('\nFixed the case of the category column, unique values now:')
    print(df['category'].unique(), '\n')

    # Process 'has_entity' column and remove rows with 'data missing'
    print('\nChecking for unique values in the has_entity column')
    print(df['has_entity'].unique())

    df = df[df['has_entity'] != 'data missing']
    print('\nRemoved rows with "data missing" in the has_entity column, unique values now:')
    print(df['has_entity'].unique())

    return df

'''
	- Function usage: df = process_df(df)
	- Replace `df` with your original DataFrame.
'''

df = process_df(df)

# Splitting the "has_entity" column into three separate binary columns
df['ORG'] = df['has_entity'].str.contains('ORG_YES').astype(int)
df['PRODUCT'] = df['has_entity'].str.contains('PRODUCT_YES').astype(int)
df['PERSON'] = df['has_entity'].str.contains('PERSON_YES').astype(int)

# Displaying the first few rows to verify the changes
print(df[['has_entity', 'ORG', 'PRODUCT', 'PERSON']].head(5))

# Display the first few rows of the dfset
print(df.head())

# Display the distribution of the categories
print("\nCategory distribution:\n", df['category'].value_counts())

# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Separating the features and labels
X = df.drop(['category', 'par_id', 'lexicon_count', 'difficult_words', 'last_editor_gender'], axis=1)
y = df['category']

# Handling the `has_entity` feature
# Assuming `has_entity` is categorical and needs to be converted into a numerical format
X = pd.get_dummies(X, columns=['has_entity'])

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Define the SMOTE and Random Under-Sampling strategy
over = SMOTE(sampling_strategy='auto')
under = RandomUnderSampler(sampling_strategy='auto')

# Create a pipeline that first applies SMOTE and then applies Random Under-Sampling
pipeline = Pipeline(steps=[('o', over), ('u', under)])

import gensim
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

download_nltk_resources()
print('\n')


# Initialize the Porter Stemmer
stemmer = PorterStemmer()
# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert NLTK's POS tags to WordNet's POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Load the Word2Vec model
file_path = './WordEmbeddings/GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

# Function to Clean, Stem, Lemmatize, and Vectorize Text
def process_text(text):
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    size = '5'
    print_size = slice(int(size))

    # Tokenize
    tokens = nltk.word_tokenize(text)
    print('Tokenized: ', tokens[print_size], '\n')
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
    print('Stopwords: ', tokens[print_size], '\n')

    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    print('Setmmed: ', stemmed_tokens[print_size], '\n')

    # Lemmatization with POS tagging
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos) or wordnet.NOUN) 
                         for token, pos in pos_tags]
    print('Lemmatized: ', lemmatized_tokens[print_size], '\n')

    # Combine stemming and lemmatization effects (optional, based on experimentation)
    final_tokens = lemmatized_tokens  # or stemmed_tokens, or a combination based on your preference
    print('Final tokens: ', final_tokens[print_size], '\n')

    # Vectorization
    valid_tokens = [word for word in final_tokens if word in word2vec_model]
    print('Valid tokens: ', valid_tokens[print_size], '\n')
    if valid_tokens:
        vector = np.mean(word2vec_model[valid_tokens], axis=0)
    else:
        vector = np.zeros(300)  # Assuming Word2Vec vectors are of size 300
    
    return vector

# This line processes the first paragraph as an example
vectorized_paragraph = process_text(df['paragraph'][0])

print("Vectorized paragraph:", vectorized_paragraph)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Step 1: Text Vectorization
tfidf = TfidfVectorizer(max_features=1000)  # Limiting to 1000 features for simplicity
X_text = tfidf.fit_transform(df['paragraph']).toarray()

# Step 2: One-Hot Encoding for `has_entity`
# Correcting the OneHotEncoder usage
encoder = OneHotEncoder()
X_entity = encoder.fit_transform(df[['has_entity']]).toarray()  # Converting to dense array immediately


# Combine text vectors and `has_entity` features
import numpy as np
X_combined = np.hstack((X_text, X_entity))

# Encode the labels
y_encoded = label_encoder.fit_transform(df['category'])

# Now split your df into a training set and a testing set to avoid df leakage
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE and Random Under-Sampling on the training set
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

# Assuming y_resampled is your array of resampled labels after applying SMOTE and under-sampling
y_resampled_labels = label_encoder.inverse_transform(y_resampled)

import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_distributions(y_initial, y_resampled):
    # Original distribution - convert y_initial to labels if it's encoded
    original_dist = pd.Series(y_initial).value_counts(normalize=True)
    
    # Adjusting for the resampled distribution - directly use y_resampled since it's already in label form
    resampled_dist = pd.Series(y_resampled).value_counts(normalize=True)
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    # Adjusting labels for original distribution if necessary
    original_labels = label_encoder.inverse_transform(original_dist.index) if original_dist.index.dtype == 'int' else original_dist.index
    
    # Pie chart before balancing
    ax[0, 0].pie(original_dist, labels=original_labels, autopct='%1.1f%%')
    ax[0, 0].set_title('Original Distribution (Pie Chart)')
    
    # Pie chart after balancing - Note: y_resampled is already in the correct format
    ax[0, 1].pie(resampled_dist, labels=resampled_dist.index, autopct='%1.1f%%')
    ax[0, 1].set_title('Balanced Distribution (Pie Chart)')
    
    # Bar chart before balancing
    ax[1, 0].bar(range(len(original_dist)), original_dist.values, tick_label=original_labels)
    ax[1, 0].set_title('Original Distribution (Bar Chart)')
    ax[1, 0].set_xticklabels(original_dist.index, rotation=45, ha="right")
    
    # Bar chart after balancing
    ax[1, 1].bar(range(len(resampled_dist)), resampled_dist.values, tick_label=resampled_dist.index)
    ax[1, 1].set_title('Balanced Distribution (Bar Chart)')
    ax[1, 1].set_xticklabels(resampled_dist.index, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.show()

# Make sure to convert y_train back to labels if it's encoded before passing
y_train_labels = label_encoder.inverse_transform(y_train) if np.issubdtype(y_train.dtype, np.integer) else y_train

# Now, call the plotting function with the correct variables
plot_distributions(y_train_labels, y_resampled_labels)

# 
X_resampled_svm_clf = X_resampled
y_resampled_svm_clf = y_resampled

# 
X_resampled_rf_clf = X_resampled
y_resampled_rf_clf = y_resampled

# 
X_resampled_mlp_clf = X_resampled
y_resampled_mlp_clf = y_resampled

# 
X_resampled_mnb_clf = X_resampled
y_resampled_mnb_clf = y_resampled

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Support Vector Classifier
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_resampled_svm_clf, y_resampled_svm_clf)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_resampled_rf_clf, y_resampled_rf_clf)

# Neural Network Classifier (MLP)
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500)
mlp_clf.fit(X_resampled_mlp_clf, y_resampled_mlp_clf)

# Multinomial Naive Bayes Classifier
mnb_clf = MultinomialNB()
mnb_clf.fit(X_resampled_mnb_clf, y_resampled_mnb_clf)

# Predict on the test set with SVM
y_pred_svm = svm_clf.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))

# Predict on the test set with Random Forest
y_pred_rf = rf_clf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# Predict on the test set with Neural Network Classifier (MLP)
y_pred_mlp = mlp_clf.predict(X_test)
print(f"Neural Network Classifier (MLP) Accuracy: {accuracy_score(y_test, y_pred_mlp)}")
print(classification_report(y_test, y_pred_mlp))

# Predict on the test set with Multinomial Naive Bayes
y_pred_mnb = mnb_clf.predict(X_test)
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_mnb)}")
print(classification_report(y_test, y_pred_mnb))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Support Vector Classifier
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_resampled_svm_clf, y_resampled_svm_clf)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_resampled_rf_clf, y_resampled_rf_clf)

# Neural Network Classifier (MLP)
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500)
mlp_clf.fit(X_resampled_mlp_clf, y_resampled_mlp_clf)

# Multinomial Naive Bayes Classifier
mnb_clf = MultinomialNB()
mnb_clf.fit(X_resampled_mnb_clf, y_resampled_mnb_clf)

from sklearn.model_selection import cross_val_score

# Perform cross-validation for each classifier
cv_scores_svm = cross_val_score(svm_clf, X_resampled_svm_clf, y_resampled_svm_clf, cv=5)
cv_scores_rf = cross_val_score(rf_clf, X_resampled_rf_clf, y_resampled_rf_clf, cv=5)
cv_scores_mlp = cross_val_score(mlp_clf, X_resampled_mlp_clf, y_resampled_mlp_clf, cv=5)
cv_scores_mnb = cross_val_score(mnb_clf, X_resampled_mnb_clf, y_resampled_mnb_clf, cv=5)

# cv_scores_svm = cross_val_score(svm_clf, X_resampled, y_resampled, cv=5)
# cv_scores_rf = cross_val_score(rf_clf, X_resampled, y_resampled, cv=5)
# cv_scores_mlp = cross_val_score(mlp_clf, X_resampled, y_resampled, cv=5)
# cv_scores_mnb = cross_val_score(mnb_clf, X_resampled, y_resampled, cv=5)

# Output the cross-validation scores for each classifier

print(f"--- Support Vector Classifier (SVM) ---")
print(f"CV scores: {cv_scores_svm}")
print(f"CV average score: {np.mean(cv_scores_svm)}")

print(f"\n--- Random Forest Classifier (RF) ---")
print(f"CV scores: {cv_scores_rf}")
print(f"CV average score: {np.mean(cv_scores_rf)}")

print(f"\n--- Neural Network Classifier (MLP) ---")
print(f"CV scores: {cv_scores_mlp}")
print(f"CV average score: {np.mean(cv_scores_mlp)}")

print(f"\n--- Multinomial Naive Bayes (MNB) ---")
print(f"CV scores: {cv_scores_mnb}")
print(f"CV average score: {np.mean(cv_scores_mnb)}")