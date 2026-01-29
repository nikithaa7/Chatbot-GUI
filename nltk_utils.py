import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
import os

# Ensure NLTK resources are downloaded
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

stemmer = PorterStemmer()

# Tokenize a sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stem a word
def stem(word):
    return stemmer.stem(word.lower())

# Create bag of words
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
