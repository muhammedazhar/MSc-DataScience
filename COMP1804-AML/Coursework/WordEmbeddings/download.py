import gensim
from gensim.models import KeyedVectors

# Example: Downloading GoogleNews-vectors-negative300.bin.gz (300-dimensional Word2Vec on Google News)
model_name = "GoogleNews-vectors-negative300.bin.gz"
model_path = f"https://s3.amazonaws.com/dl4j-distribution/{model_name}"

# Load the model
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Access word vectors using model["word"]
word_vector = model["king"]  # Get the vector for "king"