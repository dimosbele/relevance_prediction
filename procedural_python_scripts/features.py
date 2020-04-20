import utils
import stringdist
from fuzzywuzzy import fuzz


# In this file we create the features that will be used in the modeling phase
# So, we first read the files that we created in the preprocessing phase
# and we generate 3 dataframes with the following features:
# 1) df_train - it has several numeric features
# 2) df_similarities - it has features based on similarities (Levenshtein, Cosine, Jaccard, Jaro)
# 3) df_fuzzy - it has features based on this --> https://towardsdatascience.com/natural-language-processing-for-fuzzy-string-matching-with-python-6632b7824c49


def getFeatures(df_train, df_descr, df_attr):



    #### -- Features from TRAINSET + Descriptions + Attributes
    # merge the above dataframes
    df_train = df_train.merge(df_descr, left_on='product_uid', right_on='product_uid', how='left')
    df_train = df_train.merge(df_attr, left_on='product_uid', right_on='product_uid', how='left')




    #### -- General counts about numerics and non numerics in Product title and Search term
    # number of numeric terms in product_title
    df_train['N_numerics_PT'] = df_train['PT_numerics'].apply(lambda x: len(x))

    # number of numeric terms in search_term
    df_train['N_numerics_ST'] = df_train['ST_numerics'].apply(lambda x: len(x))

    # number of non numeric terms in product_title
    df_train['N_non_numerics_PT'] = df_train['PT_Non_numerics'].apply(lambda x: len(x))

    # number of non numeric terms in search_term
    df_train['N_non_numerics_ST'] = df_train['ST_Non_numerics'].apply(lambda x: len(x))

    # common nonnumeric terms between 'PT_Non_numerics' & 'ST_Non_numerics' with levensthein distance
    df_train['Common_words_leven'] = df_train.apply(lambda x: utils.common_words_leven(x['PT_Non_numerics'], x['ST_Non_numerics']), axis=1)




    #### -- Common terms between Search term & Product title
    # number of common nonnumeric terms between 'PT_Non_numerics' & 'ST_Non_numerics' with levensthein distance
    df_train['N_common_words_leven'] = df_train['Common_words_leven'].apply(lambda x: len(x))

    # Jaccard similarity based on the above common words with levensthein
    df_train['JC_sim'] = df_train.apply(lambda x: utils.get_jaccard_sim(x['PT_Non_numerics'], x['ST_Non_numerics'], x['Common_words_leven']), axis=1)




    #### -- Non numeric terms of Search term that are substrings of Product title | Descrtiption | Attribute
    ## -- PRODUCT TITLE
    # list of terms of search_term_tokens that are substrings of PT_lower
    df_train['Substrs_PT_x'] = df_train.apply(lambda x: utils.n_substrings(x['ST_Non_numerics'], x['PT_lower']), axis=1)

    # Number of terms of search_term_tokens that are substrings of product_title_lower
    df_train['N_substrs_PT_x'] = df_train['Substrs_PT_x'].apply(lambda x: len(x))


    ## -- PRODUCT DESCRIPTION
    # list of terms of search_term_tokens that are substrings of PD_lower
    df_train['Substrs_PD_x'] = df_train.apply(lambda x: utils.n_substrings(x['ST_Non_numerics'], x['PD_lower']), axis=1)

    # Number of terms of search_term_tokens that are substrings of product_title_lower
    df_train['N_substrs_PD_x'] = df_train['Substrs_PD_x'].apply(lambda x: len(x))


    ## -- PRODUCT ATTRIBUTES
    # list of terms of search_term_tokens that are substrings of PD_lower
    df_train['Substrs_Atr_x'] = df_train.apply(lambda x: utils.n_substrings(x['ST_Non_numerics'], x['Atrr_text']), axis=1)

    # Number of terms of search_term_tokens that are substrings of product_title_lower
    df_train['N_substrs_Atr_x'] = df_train['Substrs_Atr_x'].apply(lambda x: len(x))

    # percentage of terms of search_term_tokens that are substrings of PT_lower or PD_lower or Atrr_text
    df_train['Perc_substrs_x'] = df_train.apply(lambda x: utils.perc_xxx(x['Substrs_PT_x'], x['Substrs_PD_x'], x['Substrs_Atr_x'], x['ST_Non_numerics']), axis=1)


    ## -- PRODUCT TITLE
    # list of terms of search_term_tokens that are substrings of PT_lower
    df_train['Substrs_PT_y'] = df_train.apply(lambda x: utils.n_substrings(x['ST_numerics'], x['PT_lower']), axis=1)

    # Number of terms of search_term_tokens that are substrings of product_title_lower
    df_train['N_substrs_PT_y'] = df_train['Substrs_PT_y'].apply(lambda x: len(x))


    ## -- PRODUCT DESCRIPTION
    # list of terms of search_term_tokens that are substrings of PD_lower
    df_train['Substrs_PD_y'] = df_train.apply(lambda x: utils.n_substrings(x['ST_numerics'], x['PD_lower']), axis=1)

    # Number of terms of search_term_tokens that are substrings of product_title_lower
    df_train['N_substrs_PD_y'] = df_train['Substrs_PD_y'].apply(lambda x: len(x))


    ## -- PRODUCT ATTRIBUTES
    # list of terms of search_term_tokens that are substrings of PD_lower
    df_train['Substrs_Atr_y'] = df_train.apply(lambda x: utils.n_substrings(x['ST_numerics'], x['Atrr_text']), axis=1)

    # Number of terms of search_term_tokens that are substrings of product_title_lower
    df_train['N_substrs_Atr_y'] = df_train['Substrs_Atr_y'].apply(lambda x: len(x))

    # percentage of terms of search_term_tokens that are substrings of PT_lower or PD_lower or Atrr_text
    df_train['Perc_substrs_y'] = df_train.apply( lambda x: utils.perc_xxx(x['Substrs_PT_y'], x['Substrs_PD_y'], x['Substrs_Atr_y'], x['ST_numerics']), axis=1)




    #### -- Levensthein distance between Search terms & Product title"""
    # Levensthein distance between 'product_title_text' & 'search_term_text'
    df_train['Leven_sim_ST_PT'] = df_train.apply(lambda x: 1 - stringdist.levenshtein_norm(x['PT_text'], x['ST_text']), axis=1)




    #### -- Keywords of Description that appear in the Search term (with levensthein distance)
    # list of Descripton Keywords that appear in the 'ST_Non_numerics' with levensthein distance
    df_train['Keywords_leven'] = df_train.apply(lambda x: utils.common_words_leven(x['Keywords_Descr'], x['ST_Non_numerics']), axis=1)

    # number of Descripton Keywords that appear in the 'ST_Non_numerics' with levensthein distance
    df_train['N_keywords_leven'] = df_train['Keywords_leven'].apply(lambda x: len(x))

    # keep only those columns
    df_train2 = df_train[
        ['product_uid',
         'N_numerics_PT',
         'N_numerics_ST',
         'N_non_numerics_PT',
         'N_non_numerics_ST',
         'N_common_words_leven',
         'JC_sim',
         'N_substrs_PT_x',
         'N_substrs_PD_x',
         'N_substrs_Atr_x',
         'Perc_substrs_x',
         'N_substrs_PT_y',
         'N_substrs_PD_y',
         'N_substrs_Atr_y',
         'Perc_substrs_y',
         'Leven_sim_ST_PT',
         'N_keywords_leven',
         'relevance'
         ]
    ]

    return df_train, df_train2



def similarityMetrics(df_train):

    #### -- Distances: Levenshtein, Cosine, Jaccard, Jaro
    ## -- Search term vs Product Title
    # Levensthein distance between 'PT_text' & 'ST_text'
    df_train['Leven_sim_PT'] = df_train.apply(lambda x: 1 - stringdist.levenshtein_norm(x['PT_text'], x['ST_text']), axis=1)

    df_train['JC_sim_PT'] = df_train.apply(lambda x: utils.get_jaccard_sim(x['PT_stem'], x['ST_stem']), axis=1)

    # cosine
    df_train['Cosine_sim_PT'] = df_train.apply(lambda x: utils.get_cosine_sim(' '.join(x['PT_stem']), ' '.join(x['ST_stem'])), axis=1)


    ## -- Search term vs Description
    # Levensthein distance between 'Keywords_Descr' as text & 'ST_text'
    df_train['Leven_sim_PD'] = df_train.apply(lambda x: 1 - stringdist.levenshtein_norm(' '.join(x['Keywords_Descr']), x['ST_text']), axis=1)

    # Jaccard similarity
    df_train['JC_sim_PD'] = df_train.apply(lambda x: utils.get_jaccard_sim(x['PD_stem'], x['ST_stem']), axis=1)

    df_train['Cosine_sim_PD'] = df_train.apply(lambda x: utils.get_cosine_sim(' '.join(x['PD_stem']), ' '.join(x['ST_stem'])), axis=1)


    ## -- Search term vs Attributes
    df_train['Atrr_stem'] = df_train['Atrr_stem'].apply(lambda d: d if isinstance(d, list) else [])

    # Levensthein distance between 'PT_text' & 'ST_text'
    df_train['Leven_sim_Atrr'] = df_train.apply(lambda x: utils.get_leven(x), axis=1)

    # Jaccard similarity
    df_train['JC_sim_Atrr'] = df_train.apply(lambda x: utils.get_jaccard_sim(x['Atrr_stem'], x['ST_stem']), axis=1)

    # Cosine similarity
    df_train['Cosine_sim_Atrr'] = df_train.apply(lambda x: utils.get_cosine_sim(' '.join(x['Atrr_stem']), ' '.join(x['ST_stem'])), axis=1)
    df_train_sims = df_train[['id', 'product_uid', 'JC_sim_PT', 'Cosine_sim_PT', 'Leven_sim_PD', 'JC_sim_PD', 'Cosine_sim_PD', 'JC_sim_Atrr', 'Cosine_sim_Atrr', 'Leven_sim_Atrr']]
    df_similarities = df_train_sims

    return df_train, df_similarities


def fuzzy(df_train):
    #### -- Fuzzy matching
    # product title & search_term
    df_train['FZ_PT_1'] = df_train.apply(lambda x: fuzz.ratio(x['PT_lower'], x['ST_lower']), axis=1)
    df_train['FZ_PT_2'] = df_train.apply(lambda x: fuzz.partial_ratio(x['PT_lower'], x['ST_lower']), axis=1)
    df_train['FZ_PT_3'] = df_train.apply(lambda x: fuzz.token_sort_ratio(x['PT_lower'], x['ST_lower']), axis=1)
    df_train['FZ_PT_4'] = df_train.apply(lambda x: fuzz.token_set_ratio(x['PT_lower'], x['ST_lower']), axis=1)

    # convert 'Atrr_text_all' column to strings in order to avoid errors
    df_train['Atrr_text_all'] = df_train['Atrr_text_all'].astype('str')

    # product title & search_term
    df_train['FZ_Attr_1'] = df_train.apply(lambda x: fuzz.ratio(x['Atrr_text_all'], x['ST_lower']), axis=1)
    df_train['FZ_Attr_2'] = df_train.apply(lambda x: fuzz.partial_ratio(x['Atrr_text_all'], x['ST_lower']), axis=1)
    df_train['FZ_Attr_3'] = df_train.apply(lambda x: fuzz.token_sort_ratio(x['Atrr_text_all'], x['ST_lower']), axis=1)
    df_train['FZ_Attr_4'] = df_train.apply(lambda x: fuzz.token_set_ratio(x['Atrr_text_all'], x['ST_lower']), axis=1)

    df_fuzzy = df_train[['product_uid', 'FZ_PT_1', 'FZ_PT_2', 'FZ_PT_3', 'FZ_PT_4', 'FZ_Attr_1',
                         'FZ_Attr_2', 'FZ_Attr_3', 'FZ_Attr_4']]

    return df_fuzzy