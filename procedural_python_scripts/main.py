import pandas as pd
import utils
import preprocessing
import features
import modeling

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)


df_train = utils.getURLtoDF('http://bitbucket.org/dimitrisor/nlp_coursework_relevance_score_prediction/raw/4b81408f87a4f72746c45d47c5c385437df1ec5b/train.csv')
df_descr = utils.getURLtoDF('http://bitbucket.org/dimitrisor/nlp_coursework_relevance_score_prediction/raw/4b81408f87a4f72746c45d47c5c385437df1ec5b/product_descriptions.csv')
df_attr = utils.getURLtoDF('http://bitbucket.org/dimitrisor/nlp_coursework_relevance_score_prediction/raw/4b81408f87a4f72746c45d47c5c385437df1ec5b/attributes.csv')


# Phase 1 - Preproccessing

df_train = preprocessing.prep_trainset(df_train) # Trainset preproccessing
df_descr = preprocessing.prep_descr(df_descr, df_train) # Product descriptions preproccessing
df_attr = preprocessing.prep_attr(df_attr, df_train) # Product attributes preproccessing

# df_train.to_pickle('df_train_prep.pkl')
# df_descr.to_pickle('df_descr_prep.pkl')
# df_attr.to_pickle('df_attr_prep_new1.pkl')

# df_train = pd.read_pickle('df_train_prep.pkl')
# df_descr = pd.read_pickle('df_descr_prep.pkl')
# df_attr = pd.read_pickle('df_attr_prep.pkl')


# Phase 2 Feature enginnering
# df_train, df_similarities, df_fuzzy = features.feature_engineering(df_train, df_descr, df_attr)

df_train, df_train2 = features.getFeatures(df_train, df_descr, df_attr) # df_train2 stored for modelling phase
df_train, df_similarities = features.similarityMetrics(df_train) #
df_fuzzy = features.fuzzy(df_train)

# df_train.to_pickle('df_train_feat.pkl')
# df_similarities.to_pickle('df_similarities_feat.pkl')
# df_fuzzy.to_pickle('df_fuzzy_feat.pkl')
# df_train = pd.read_pickle('df_train_feat.pkl')
# df_similarities = pd.read_pickle('df_similarities_feat.pkl')
# df_fuzzy = pd.read_pickle('df_fuzzy_feat.pkl')

# Phase 3 Modelling

modeling.run(df_train2, df_similarities, df_fuzzy)