from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.text_preprocessing import tokenize

def binary_vectorizer(texts):
    """
    Converts a list of texts to binary vectors.
    """
    vectorizer = CountVectorizer(binary=True)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer

def count_vectorizer(texts):
    """
    Converts a list of texts to count vectors.
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer

def tfidf_vectorizer(texts):
    """
    Converts a list of texts to TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer

