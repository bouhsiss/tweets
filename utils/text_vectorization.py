from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from text_preprocessing import tokenize

def binary_vectorizer(texts):
    """
    Converts a list of texts to binary vectors.
    """
    vectorizer = CountVectorizer(binary=True)
    return vectorizer.fit_transform(texts)

def count_vectorizer(texts):
    """
    Converts a list of texts to count vectors.
    """
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(texts)

def tfidf_vectorizer(texts):
    """
    Converts a list of texts to TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

