import string
import nltk

from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tokens(text):
    # Tokenize text into words
    lower = text.lower()
    remove_punctuation_map = dict(
        (ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stem_tokens(tokens, stemmer):
    # Apply stemming to a list of tokens
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def get_cosine_sim(i, j, mat_tf):
    # Calculate cosine similarity between two vectors in the matrix
    return mat_tf[i, j]


def has_question(sentence):
    # Check if a sentence contains a question mark
    return "?" in sentence


def normalize(text):
    # Normalize text by tokenizing and stemming
    return stem_tokens(get_tokens(text), PorterStemmer())


def get_tfidf_sim(df):
    # Get the TF-IDF cosine similarity matrix for a DataFrame
    document = list(df['utterance'])
    return get_tfidf_cosine(document)


def get_tfidf_cosine(document):
    # Calculate the TF-IDF cosine similarity matrix for a document
    vect = TfidfVectorizer(tokenizer=normalize, min_df=1)
    tfidf = vect.fit_transform(document)
    pairwise_similarity = (tfidf * tfidf.T).toarray()
    return pairwise_similarity
