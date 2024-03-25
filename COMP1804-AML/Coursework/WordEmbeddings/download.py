import os
import logging
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

print("Downloading Word2Vec model...")
w2v_model = api.load('word2vec-google-news-300')

print("Downloading SentenceTransformer model...")
MiniLM_model = SentenceTransformer('all-MiniLM-L6-v2')

# Save the models for later use
print("Saving word2vec model...")
w2v_model.save(os.path.join(script_dir, "word2vec.model"))

print("Saving SentenceTransformer model...")
MiniLM_model.save(os.path.join(script_dir, 'MiniLM.model'))

print("Done.")