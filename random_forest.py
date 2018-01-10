import pandas as pd
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

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
train, test = utils.one_hot_encode_categorical(train, test)

# Remove constant features
desc = train.describe()
feat_to_drop = [c for c in desc.columns if desc[c][2] == 0]
train.drop(feat_to_drop, axis = 1, inplace = True)
test.drop(feat_to_drop, axis = 1, inplace = True)

y = train['y']
test_ids = test['ID']
test.drop('ID', axis = 1, inplace = True)
train.drop(['ID', 'y'], axis = 1, inplace = True)

n_components = 12
# PCA
df_pca, df_test_pca = utils.pca(train, test, n_components)

# ICA
columns = ['ICA_{}'.format(i) for i in range(n_components)]
ica = FastICA(n_components=n_components, random_state = 42, max_iter = 10000, tol = 0.001)
df_ica = pd.DataFrame(ica.fit_transform(train), columns = columns)
df_test_ica = pd.DataFrame(ica.transform(test), columns = columns)

# Truncated SVD
columns = ['TSVD_{}'.format(i) for i in range(n_components)]
tsvd = TruncatedSVD(n_components=n_components, random_state=420)
df_tsvd = pd.DataFrame(tsvd.fit_transform(train), columns = columns)
df_test_tsvd = pd.DataFrame(tsvd.transform(test), columns = columns)

# GRP
columns = ['GRP_{}'.format(i) for i in range(n_components)]
grp = GaussianRandomProjection(n_components=n_components, eps=0.1, random_state=420)
df_grp = pd.DataFrame(grp.fit_transform(train), columns = columns)
df_test_grp = pd.DataFrame(grp.transform(test), columns = columns)

# SRP
columns = ['SRP_{}'.format(i) for i in range(n_components)]
srp = SparseRandomProjection(n_components=n_components, dense_output=True, random_state=420)
df_srp = pd.DataFrame(srp.fit_transform(train), columns = columns)
df_test_srp = pd.DataFrame(srp.transform(test), columns = columns)


train = pd.concat([train, df_pca, df_ica, df_tsvd, df_grp, df_srp], axis = 1)
test = pd.concat([test, df_test_pca, df_test_ica, df_test_tsvd, df_test_grp, df_test_srp], axis = 1)



### Grid Search
rf = RandomForestRegressor(n_estimators=1000)
rf.fit(train, y)
params = {'n_estimators': [500, 1000]}
utils.grid_search(rf, params, train, y)