import pandas as pd
import utils
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Analisi y e outliers
plt.boxplot(train.y)
plt.show()
out = train[train.y > 250]
train.drop(out.index, axis = 0, inplace = True)
train.reset_index(drop = True, inplace = True)
del out
plt.boxplot(train.y)
plt.show()
plt.hist(train.y, bins = 20)
plt.show()


# Normalize y
scaler = StandardScaler()
scaler = scaler.fit(train.y)
y_norm = pd.Series(scaler.transform(train.y))
train.y = y_norm
# Plot again y
plt.boxplot(train.y)
plt.show()
plt.hist(train.y, bins = 20)
plt.show()

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
#df_categorical = train[categorical_columns]
#train, test = utils.one_hot_encode_categorical(train, test)

# Label encoding
train, test = utils.label_encode_categorical(train, test)

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
ica = FastICA(n_components=n_components, random_state = 420, max_iter = 10000, tol = 0.001)
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

### FEATURE SELECTION ###
# f_regression
#f_sel = SelectKBest(score_func = f_regression, k = 'all')
#train_red = pd.DataFrame(f_sel.fit_transform(train, y))
#f_scores = pd.Series(f_sel.scores_)
#pvalues = pd.Series(f_sel.pvalues_)
#test_red = pd.DataFrame(f_sel.transform(test))
# mutual_info
#sel = SelectKBest(score_func = mutual_info_regression, k = 'all')
#sel.fit(train, y)
#m_scores = pd.Series(sel.scores_)
#m_pvalues = pd.Series(sel.pvalues_)
#best_30 = [train.columns[i] for i in m_scores.sort_values(ascending = False)[:30].index]

train_orig = train
test_orig = test
#train = train.loc[:, best_30]
#test = test.loc[:, best_30]


#==============================================================================
# ### Prova Ridge
# ridge = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 5, 10, 50, 100], cv = 10)
# ridge.fit(train, y)
# r = Ridge(alpha=10)
# cross = cross_val_score(r, train, y, scoring = 'r2', cv = 10)
# print(cross.mean())
# cross
# r.fit(train, y)
# y_pred = r.predict(test)
#==============================================================================

# Lasso e RFECV
lasso = LassoCV()
rfe = RFECV(lasso, cv = 3, n_jobs=-1, verbose = True)
rfe.fit(train, y)
supp = pd.Series(rfe.support_)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (R2)")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()
print(len(supp[supp == True]))
print(supp[supp == True].index)
lasso = LassoCV(cv = 10)
x = train[['X47', 'X0_mean', 'X5_mean']]
lasso.fit(x, y)
l = Lasso(alpha=lasso.alpha_)
cross = cross_val_score(l, x, y, scoring='r2', cv = 20)
print(cross.mean())
cross
l.fit(x, y)
y_pred = l.predict(test[['X0_mean', 'X5_mean', 'X47']])


def grid_search(estimator, params, train):
    gsearch1 = GridSearchCV(estimator = estimator, 
                        param_grid = params, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
    gsearch1.fit(train,y)
    return gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

### SVR
svr = SVR(kernel='linear')
params = {'C': [i/10 for i in range(1, 22, 5)],
          'epsilon': [i/10 for i in range(1, 11, 2)],
          'gamma': [i/10 for i in range(1, 10, 2)]}
grid_search(svr, params, x)
#{'C': 1.6, 'epsilon': 0.5, 'gamma': 0.1}, 0.57872483401522012)


params_2 = {'C': [i/10 for i in range(12, 22)],
          'epsilon': [i/10 for i in range(4, 7)],
          'gamma': [i/100 for i in range(1, 10, 2)]}
grid_search(svr, params_2)
#{'C': 2.1, 'epsilon': 0.4, 'gamma': 0.01}, 0.5917524195649051)

params_3 = {'C': [i/10 for i in range(21, 30)],
          'epsilon': [i/10 for i in range(4, 7)],
          'gamma': [i/100 for i in range(1, 10, 2)]}
grid_search(svr, params_3)
#{'C': 2.5, 'epsilon': 0.4, 'gamma': 0.01}, 0.59183928374500483)


params_4 = {'C': [i/10 for i in range(25, 26)],
          'epsilon': [i/10 for i in range(4, 5)],
          'gamma': [i/1000 for i in range(1, 10, 2)]}
grid_search(svr, params_4)

svr = SVR(kernel = 'linear', gamma = 0.0001, epsilon=0.5, C = 1.6)
cross = cross_val_score(svr, x, y, cv = 20, n_jobs=-1, scoring='r2')
print(cross.mean())
cross
svr.fit(x, y)
y_pred = svr.predict(test[['X0_mean', 'X5_mean', 'X47']])




# Denormalize the predictions
y_pred_denorm = scaler.inverse_transform(y_pred)
sub = pd.DataFrame(data = {'ID': test_ids, 'y': y_pred_denorm})
sub.to_csv('submissions/svr_g0.0001_eps_0.5_c1.6_3feat.csv', index = False)