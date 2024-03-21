# Pre-processing guide

**Step 1: Text pre-processing for the 'category' prediction task:**

The 'paragraph' column in the dataset will be used to predict the 'category'. However, machine learning models don't understand raw text data, so the text data must be transformed into numerical features that the models can understand. 

For text data, you can use techniques such as:

- Removing punctuation and transforming all text to lower case.
- Tokenization.
- Removing stop words.
- Stemming/Lemmatization.
- Vectorization (Bag of words or TF-IDF).

I recommend using the Natural Language Toolkit (NLTK) for most of these tasks, and sklearn.feature_extraction.text.TfidfVectorizer() for vectorization.


**Step 2: Handling the 'text_clarity' task:**

First, manually label a subset of the data to have 'clear_enough' and 'not_clear_enough' classes. For this task, you might want to consider sentences with complex words, longer length, or difficult readability score as 'not_clear_enough'. This is subjective and would need your judgement.

After creating a labeled subset for the 'text_clarity' task, we need to prepare this text data for model building. You can follow the same steps as Step 1 for this.


After pre-processing, you can divide the modified dataset into training and testing datasets. Randomly selecting 80% of the data for training and 20% for testing should provide a good start.

Once all of this is done, please upload your prepared datasets to the workspace and we can move on to the next step!