
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import utils


def prep_trainset(df_train):

    # -- Column product_title - PT"""

    # convert to lower case
    df_train['PT_lower'] = df_train['product_title'].apply(lambda text: text.lower())

    # remove punctuation and tokenize
    # create a new column with the tokens
    tokenizer = RegexpTokenizer(r'\w+')
    df_train['PT_tokens'] = df_train['PT_lower'].apply(lambda text: tokenizer.tokenize(text))

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    df_train['PT_tokens_sw'] = df_train['PT_tokens'].apply(lambda tokens: [i for i in tokens if i not in stop_words])

    # create a column wiht the tokens as text (without stopwords)
    df_train['PT_text'] = df_train['PT_tokens_sw'].apply(lambda tokens: ' '.join(tokens))

    # stemming
    stemmer = PorterStemmer()
    df_train['PT_stem'] = df_train['PT_tokens_sw'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

    # create a list of terms of 'product_title' that contain numbers
    df_train['PT_numerics'] = df_train['PT_lower'].apply(lambda x: utils.hasNumbers(x))

    df_train['PT_Non_numerics'] = df_train['PT_lower'].apply(lambda x: utils.has_not_Numbers(x))

    """#### -- Column search_term - ST"""

    # convert to lower case
    df_train['ST_lower'] = df_train['search_term'].apply(lambda text: text.lower())

    # remove punctuation and tokenize
    # create a new column with the tokens
    tokenizer = RegexpTokenizer(r'\w+')
    df_train['ST_tokens'] = df_train['ST_lower'].apply(lambda text: tokenizer.tokenize(text))

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    df_train['ST_tokens_sw'] = df_train['ST_tokens'].apply(lambda tokens: [i for i in tokens if i not in stop_words])

    # create a column with the tokens as text (without stopwords)
    df_train['ST_text'] = df_train['ST_tokens_sw'].apply(lambda tokens: ' '.join(tokens))

    # stemming
    stemmer = PorterStemmer()
    df_train['ST_stem'] = df_train['ST_tokens_sw'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

    # create a list of terms of 'search_term_tokens' that contain numbers
    df_train['ST_numerics'] = df_train['ST_lower'].apply(lambda x: utils.hasNumbers(x))

    df_train['ST_Non_numerics'] = df_train['ST_lower'].apply(lambda x: utils.has_not_Numbers(x))

    print(df_train.head(20))

    return df_train



def prep_descr(df_descr, df_train):

    # In this notebook we preprocess the product_descriptions.csv and finally we save the preprocessed file as df_descr_kwd.pkl
    # Finally we save a final preprocessed file that will be used in the feature engineering : df_descr_kwd.pkl

    import pandas as pd

    # print non truncated column info in pandas dataframe
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', 500)

    from nltk.tokenize import RegexpTokenizer

    from nltk.corpus import stopwords

    from nltk.stem.porter import PorterStemmer

    ## - Preprocess product_descriptions.csv

    # keep only the description of the products that appear in the trainset
    df_descr = df_descr[df_descr.product_uid.isin(df_train.product_uid.unique())]

    df_descr.head(1)

    # convert to lower case
    df_descr['PD_lower'] = df_descr['product_description'].apply(lambda text: text.lower())

    # remove punctuation and tokenize
    # create a new column with the tokens
    tokenizer = RegexpTokenizer(r'\w+')
    df_descr['PD_tokens'] = df_descr['PD_lower'].apply(lambda text: tokenizer.tokenize(text))

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    df_descr['PD_tokens_sw'] = df_descr['PD_tokens'].apply(
        lambda tokens: [i for i in tokens if i not in stop_words])

    # create a column wiht the tokens as text (without stopwords)
    df_descr['PD_text'] = df_descr['PD_tokens_sw'].apply(lambda tokens: ' '.join(tokens))

    # stemming
    stemmer = PorterStemmer()
    df_descr['PD_stem'] = df_descr['PD_tokens_sw'].apply(
        lambda tokens: [stemmer.stem(token) for token in tokens])



    # create a list of terms of 'product_title' that contain numbers
    df_descr['PD_numerics'] = df_descr['PD_lower'].apply(lambda x: utils.hasNumbers(x))


    # create a list of terms of 'PD_tokens' that don't contain numbers
    df_descr['PD_Non_numerics'] = df_descr['PD_tokens_sw'].apply(lambda x: utils.has_not_Numbers(x))

    # create a column with the text of each product tile without any numeric terms
    df_descr['PD_Clean_text'] = df_descr['PD_Non_numerics'].apply(lambda x: ' '.join(x))

    df_descr.head(1)

    # -- TFIDF of Descriptions --> Keywords

    # we will extract 10 keywords of each product description by using tf-idf

    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = df_descr.PD_Clean_text.values.tolist()

    len(corpus)

    # create the tfidf matrix
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    idf = list(vectorizer.idf_)
    terms = vectorizer.get_feature_names()

    # create a dataframe with the terms and the tfidf score
    df_tfidf = pd.DataFrame({'Term': terms, 'Score': idf})

    # sort values
    df_tfidf = df_tfidf.sort_values(by=['Score'], ascending=False)
    df_tfidf.head()

    df_tfidf.shape

    ## --- Keywords in Product Descriptions based on TF-IDF

    def find_keywords(tokens, df, n):
        try:
            df_keywords = df[df.Term.isin(tokens)]
            df_keywords = df_keywords.head(n)
            keywords = df_keywords.Term.values.tolist()
        except:
            keywords = []

        return keywords

    # find the keywords of each description by using the df_tfidf dataframe
    df_descr['Keywords_Descr'] = df_descr.apply(
        lambda x: find_keywords(x['PD_Non_numerics'], df_tfidf, 10),
        axis=1)

    df_descr.head(15)

    # keep only 2 columns
    df_descr_keywords = df_descr[['product_uid', 'Keywords_Descr']]
    df_descr_keywords.head()

    # Merge descriptions and keywords
    df_descriptions2 = df_descr[['product_uid', 'PD_lower', 'PD_stem', 'PD_numerics']]
    df_descriptions3 = df_descriptions2.merge(df_descr_keywords, left_on='product_uid', right_on='product_uid', how='left')
    df_descriptions3.head(2)

    return df_descriptions3

def prep_attr(df_attrs, df_train):

    # keep only products that appear in the trainset
    df_attrs = df_attrs[df_attrs.product_uid.isin(df_train.product_uid.unique())]

    print(df_attrs.shape)

    print(df_attrs.product_uid.nunique())

    print(df_attrs.nunique())

    # -- Check the coverage of attributes
    # The target is to keep only attributes that appear in almost all the products.

    # just check the attributes that appear in many products
    df_attr_names = df_attrs.groupby('name').count().sort_values(by=['product_uid'], ascending=False)
    df_attr_names['Coverage'] = df_attr_names['product_uid']/df_attrs.product_uid.nunique()
    df_attr_names.head(10)

    # group all attributes for each product
    df_attrs_groups = df_attrs.groupby(['product_uid'], as_index=True)['value'].apply(list).reset_index()
    df_attrs_groups.columns = ['product_uid', 'value_all']
    df_attrs_groups.head(1)

    # group the 5 important attributes for each product
    df_attrs_5 = df_attrs[df_attrs.name.isin(['MFG Brand Name', 'Bullet02', 'Bullet03', 'Bullet04', 'Bullet01'])]
    df_attrs_groups_5 = df_attrs_5.groupby(['product_uid'], as_index=True)['value'].apply(list).reset_index()
    df_attrs_groups_5.columns = ['product_uid', 'value_5']
    df_attrs_groups_5.head(1)

    # merge the above dataframes
    df_attrs_groups = df_attrs_groups.merge(df_attrs_groups_5, left_on='product_uid', right_on='product_uid', how='left')

    df_attrs_groups.head(1)

    # merge the text of each attribute to a common text
    df_attrs_groups['Atrr_text_all'] = df_attrs_groups['value_all'].apply(lambda x: ' '.join(str(i) for i in x))
    df_attrs_groups['Atrr_text_5'] = df_attrs_groups['value_5'].apply(lambda x: ' '.join(str(i) for i in x))

    # convert to lower case
    df_attrs_groups['Atrr_text_all'] = df_attrs_groups['Atrr_text_all'].apply(lambda text: text.lower())
    df_attrs_groups['Atrr_text_5'] = df_attrs_groups['Atrr_text_5'].apply(lambda text: text.lower())

    del df_attrs_groups['value_all']
    del df_attrs_groups['value_5']

    # remove punctuation and tokenize
    # create a new column with the tokens
    tokenizer = RegexpTokenizer(r'\w+')
    df_attrs_groups['Atrr_tokens'] = df_attrs_groups['Atrr_text_5'].apply(lambda text: tokenizer.tokenize(text))

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    df_attrs_groups['Atrr_tokens_sw'] = df_attrs_groups['Atrr_tokens'].apply(lambda tokens: [i for i in tokens if i not in stop_words])

    # create a column wiht the tokens as text (without stopwords)
    df_attrs_groups['Atrr_text'] = df_attrs_groups['Atrr_tokens_sw'].apply(lambda tokens: ' '.join(tokens))

    # stemming
    stemmer = PorterStemmer()
    df_attrs_groups['Atrr_stem'] = df_attrs_groups['Atrr_tokens_sw'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

    df_attrs_groups.head(1)

    df_attrs_groups.columns

    df_attrs_groups2 = df_attrs_groups[['product_uid', 'Atrr_text_all', 'Atrr_stem', 'Atrr_text']]
    return  df_attrs_groups2