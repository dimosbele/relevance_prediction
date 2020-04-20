import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt




def run(df_train, df_similarities, df_fuzzy):

    # merge the main trainset with the extra similarities
    df = df_train.merge(df_similarities, left_on='product_uid', right_on='product_uid', how='left')
    # and fuzzy features
    df = df.merge(df_fuzzy, left_on='product_uid', right_on='product_uid', how='left')


    df.head(2)
    df.describe()
    sns.distplot(df['relevance'])
    df.corr()
    df.columns


    # df = df.drop(['Leven_sim_Atrr'], axis=1) # Drop column
    df = df.dropna(subset=['Leven_sim_Atrr'])  # Drop rows with 'Leven_sim_Atrr' nan

    X = df[['id', 'product_uid', 'N_numerics_PT', 'N_numerics_ST', 'N_non_numerics_PT',
            'N_non_numerics_ST', 'N_common_words_leven', 'JC_sim', 'N_substrs_PT_x',
            'N_substrs_PD_x', 'N_substrs_Atr_x', 'Perc_substrs_x', 'N_substrs_PT_y',
            'N_substrs_PD_y', 'N_substrs_Atr_y', 'Perc_substrs_y',
            'Leven_sim_ST_PT', 'N_keywords_leven', 'JC_sim_PT',
            'Cosine_sim_PT', 'Leven_sim_PD', 'JC_sim_PD', 'Cosine_sim_PD',
            'JC_sim_Atrr', 'Cosine_sim_Atrr', 'FZ_PT_1',
            'FZ_PT_2', 'FZ_PT_3', 'FZ_PT_4', 'FZ_Attr_1', 'FZ_Attr_2', 'FZ_Attr_3',
            'FZ_Attr_4']]

    y = df['relevance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    print(X_train.shape, X_test.shape)




    #### -- DecisionTreeRegressor
    DT = DecisionTreeRegressor(criterion='mse', max_depth=100, max_leaf_nodes=250, min_samples_leaf=10, min_samples_split=17)
    DT.fit(X_train, y_train)

    print("R-Squared on train dataset={}".format(DT.score(X_train, y_train)))
    print("R-Squared on test dataset={}".format(DT.score(X_test, y_test)))

    predictions_test = DT.predict(X_test)
    predictions_train = DT.predict(X_train)

    rms_train = sqrt(mean_squared_error(y_train, predictions_train))
    rms_test = sqrt(mean_squared_error(y_test, predictions_test))

    print('RMSE - train = ', rms_train)
    print('RMSE - test = ', rms_test)




    #### -- GridSearchCV - DecisionTreeRegressor
    param_grid = {
        "criterion": ["mse"],
        "min_samples_split": [13, 17],
        "max_depth": [80, 100],
        "min_samples_leaf": [10, 15],
        "max_leaf_nodes": [200, 250]
    }

    grid_cv_DT = GridSearchCV(DT, param_grid, cv=10)
    grid_cv_DT.fit(X_train, y_train)

    print("R-Squared::{}".format(grid_cv_DT.best_score_))
    print("Best Hyperparameters::\n{}".format(grid_cv_DT.best_params_))




    {grid_cv_DT.best_score_: grid_cv_DT.best_estimator_}

    model_DT = grid_cv_DT.best_estimator_
    model_DT.fit(X_train, y_train)

    print("R-Squared on train dataset={}".format(model_DT.score(X_train, y_train)))
    print("R-Squared on test dataset={}".format(model_DT.score(X_test, y_test)))




    predictions_test = model_DT.predict(X_test)
    predictions_train = model_DT.predict(X_train)

    rms_train = sqrt(mean_squared_error(y_train, predictions_train))
    rms_test = sqrt(mean_squared_error(y_test, predictions_test))

    print('RMSE - train = ', rms_train)
    print('RMSE - test = ', rms_test)




    summary = pd.DataFrame(model_DT.feature_importances_, index=X_train.columns, columns=['Feature Importance'])
    ax = plt.gca()
    summary.sort_values('Feature Importance').plot.bar(ax=ax, title='10 Folds Cross-validated RMSE: {0}'.format(rms_test), figsize=(10, 5))

    # save model to disk
    # filename = 'model_DT.sav'
    # pickle.dump(model_DT, open(filename, 'wb'))

    # load saved model
    # filename = 'models/model_DT.sav'
    # loaded_model = pickle.load(open(filename, 'rb'))