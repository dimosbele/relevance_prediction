import stringdist
import pandas as pd
import requests
from io import StringIO

def getURLtoDF(url):
    req = requests.get(url, verify=False)
    df = pd.read_csv(StringIO(req.text))
    return df

# create a list of terms of 'PD_tokens' that contain numbers
def hasNumbers(text):
    term_list = text.split(' ')
    numerics = []
    for term in term_list:
        if any(char.isdigit() for char in term) == 1:
            numerics.append(term)
    return numerics

# create a list of terms of 'PD_tokens' that don't contain numbers
def has_not_Numbers(terms):
    # term_list = text.split(' ')
    nonnumerics = []
    for term in terms:
        if any(char.isdigit() for char in term) == 1:
            continue
        else:
            nonnumerics.append(term)
    return nonnumerics



    # -- Common terms between Search term & Product title
    # -- Keywords of Description that appear in the Search term (with levensthein distance)
    # -- Keywords of Description that appear in the Search term (with levensthein distance)

def common_words_leven(tokens_1, tokens_2):
    # N = 0
    common_terms = []
    tokens_1 = list(set(tokens_1))
    tokens_2 = list(set(tokens_2))

    for token1 in tokens_1:
        for token2 in tokens_2:
            if 1 - stringdist.levenshtein_norm(token1, token2) > 0.85:
                # N += 1
                common_terms.append(token2)
    try:
        return common_terms
    except:
        return common_terms


# Jaccard similarity
def get_jaccard_sim(words1, words2):
    """Returns jaccard similarity between 2 list of words"""
    try:
        a = set(words1)
        b = set(words2)
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return 0


def perc_xxx(tokens1, tokens2, tokens3, tokens4):
    try:
        tokens123 = list(set(tokens1 + tokens2 + tokens3))
        return len(tokens123) / len(tokens4)
    except:
        return 0




def get_cosine_sim(*strs):
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        vectors = [t for t in get_vectors(*strs)]
        return cosine_similarity(vectors)[0][1]
    except:
        return 0




# Νumber of non numeric terms of the search_term that appears in the Product title | Descrtiption | Attributes
def n_substrings(tokens, text):
    # N = 0
    substrings = []
    for token in tokens:
        try:
            if token in text:
                # N += 1
                substrings.append(token)
        except:
            # in case the text is float
            pass
    return substrings



def get_vectors(*strs):
    from sklearn.feature_extraction.text import CountVectorizer
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()



def get_leven(x):
    try:
        return 1 - stringdist.levenshtein_norm(' '.join(x['Atrr_stem']), x['ST_text'])
    except:
        # print('error')
        return 0


# number of numeric terms of the search_term that appears in the Product title | Descrtiption | Attributes
def n_substrings(tokens, text):
    # N = 0
    substrings = []
    for token in tokens:
        try:
            if token in text:
                # N += 1
                substrings.append(token)
        except:
            # in case the text is float
            pass
    return substrings



def common_words_leven(tokens_1, tokens_2):
    # N = 0
    common_terms = []
    tokens_1 = list(set(tokens_1))
    tokens_2 = list(set(tokens_2))

    for token1 in tokens_1:
        for token2 in tokens_2:
            if 1 - stringdist.levenshtein_norm(token1, token2) > 0.85:
                # N += 1
                common_terms.append(token1)
    try:
        return common_terms
    except:
        return common_terms