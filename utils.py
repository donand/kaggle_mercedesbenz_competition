import pandas as pd
from sklearn.decomposition import PCA, FastICA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

def label_encode_categorical(df, df_test):
    categorical_columns = ['X' + str(i) for i in range(0,9) if i != 7]
    df1 = df.copy()
    df2 = df_test.copy()
    for c in categorical_columns:
        le = LabelEncoder()
        le.fit(list(df[c].values) + list(df_test[c].values))
        df1[c] = le.transform(list(df[c].values))
        df2[c] = le.transform(list(df_test[c].values))
    return df1, df2

def one_hot_encode_categorical(df, df_test):
    categorical_columns = ['X' + str(i) for i in range(0,9) if i != 7]
    for c in categorical_columns:
        categories = list(set.intersection(set(df[c].tolist()), set(df_test[c].tolist())))
        df[c] = df[c].astype('category', categories = categories)
        df_test[c] = df_test[c].astype('category', categories = categories)
    # Eliminare una colonna per ogni colonna categorical
    return pd.get_dummies(df), pd.get_dummies(df_test)

def pca(df, df_test, percentage):
    pca = PCA(n_components=percentage, svd_solver='full', random_state = 42)
    df1 = pca.fit_transform(df)
    df2 = pca.transform(df_test)
    print('PCA - number of components {}/{}'.format(pca.n_components_, len(df.columns)))
    columns = ['PCA_{}'.format(i) for i in range(pca.n_components_)]
    return pd.DataFrame(df1, columns = columns), pd.DataFrame(df2, columns = columns)


def parameter_search(df, target, model_list, value_list, scoring, xlabel = 'values', cv = 5):
    if len(value_list) != len(model_list):
        return
    values = []
    scores = []
    print('Performing paratemeter search for {}'.format(model_list[0]))
    for i in range(len(value_list)):
        n = value_list[i]
        m = model_list[i]
        print('Current value = ' + str(n))
        cross = cross_val_score(m, df, target, cv = cv, scoring = scoring)
        values.append(n)
        scores.append(cross.mean())
        print('Average score: {}'.format(cross.mean()))
    plt.plot(values, scores)
    plt.xlabel(xlabel)
    plt.ylabel(scoring)
    plt.show()
    best_value = values[np.asarray(scores).argmax()]
    best_score = np.asarray(scores).max()
    return values, scores, best_value, best_score

def generate_submission(model, df, y, df_test, test_ids, filename):
    model.fit(df, y)
    pred = model.predict(df_test)
    sub = pd.DataFrame(data = {'ID': test_ids, 'y': pred})
    sub.to_csv('submissions/{}'.format(filename), index = False)
    
    

def grid_search(estimator, params, train, y):
    gsearch1 = GridSearchCV(estimator = estimator, 
                        param_grid = params, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
    gsearch1.fit(train,y)
    return gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    
    
    
def transform_dataset(train, test):
    # Analisi y e outliers
    out = train[train.y > 250]
    train.drop(out.index, axis = 0, inplace = True)
    train.reset_index(drop = True, inplace = True)
    
    # Normalize y
    scaler = StandardScaler()
    scaler = scaler.fit(train.y)
    y_norm = pd.Series(scaler.transform(train.y))
    train.y = y_norm
    
    # Add mean and median for values of features
    categorical_columns = ['X' + str(i) for i in range(0,9) if i != 7]
    for c in categorical_columns:
        group = train[[c, 'y']].groupby(c, as_index = False).mean()
        group.columns = [c, '{}_mean'.format(c)]
        train = pd.merge(train, group, on = c, how = 'outer')
        test = pd.merge(test, group, on = c, how = 'left')
        test['{}_mean'.format(c)].fillna(test['{}_mean'.format(c)].dropna().mean(), inplace=True)
        group = train[[c, 'y']].groupby(c, as_index = False).median()
        group.columns = [c, '{}_median'.format(c)]
        train = pd.merge(train, group, on = c, how = 'outer')
        test = pd.merge(test, group, on = c, how = 'left')
        test['{}_median'.format(c)].fillna(test['{}_median'.format(c)].dropna().mean(), inplace=True)
    
    # Add frequencies to categorical
    for col in categorical_columns:    
        train[col + '_freq'] = train.groupby(col)[col].transform('value_counts')/len(train)
        test[col + '_freq'] = test.groupby(col)[col].transform('value_counts')/len(test)
    
    # Add one hot encoding
    train, test = one_hot_encode_categorical(train, test)
    
    # Remove constant features
    desc = train.describe()
    feat_to_drop = [c for c in desc.columns if desc[c][2] == 0]
    train.drop(feat_to_drop, axis = 1, inplace = True)
    test.drop(feat_to_drop, axis = 1, inplace = True)
    
    y = train['y']
    test_ids = test['ID']
    test.drop('ID', axis = 1, inplace = True)
    train.drop(['ID', 'y'], axis = 1, inplace = True)
    
    # PCA
    df_pca, df_test_pca = pca(train, test, 30)
    
    # ICA
    columns = ['ICA_{}'.format(i) for i in range(10)]
    ica = FastICA(n_components=10, random_state = 42, max_iter = 10000)
    df_ica = pd.DataFrame(ica.fit_transform(train), columns = columns)
    df_test_ica = pd.DataFrame(ica.transform(test), columns = columns)
    train = pd.concat([train, df_pca, df_ica], axis = 1)
    test = pd.concat([test, df_test_pca, df_test_ica], axis = 1)
    
    return train, test, y, test_ids, scaler
