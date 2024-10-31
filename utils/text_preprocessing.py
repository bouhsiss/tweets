import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from symspellpy import SymSpell, Verbosity


# Map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# initialize stemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    """
    Tokenizes the input text into words.
    """
    return word_tokenize(text)

def stem_text(text):
    """
    Stems each word in the input text.
    """
    tokens = tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_words)

def lemmatize_text(text):
    """
    Lemmatizes each word in the input text.
    """
    tokens = tokenize(text)

    pos_tags = pos_tag(tokens)
    
    lemmatized_words = []
    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return " ".join(lemmatized_words)


# initialize SymSpell to correct spelling
sym_spell = SymSpell()
sym_spell.load_dictionary('frequency_dictionary_en_82_765.txt',\
                      term_index=0, \
                      count_index=1, \
                      separator=' ')

def correct_spelling(text):
    """
    Corrects spelling in the input text using SymSpell.
    """
    corrected_words = []
    for word in text.split():
        suggestion = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestion[0].term if suggestion else word)
    return " ".join(corrected_words)
