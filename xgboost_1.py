import pandas as pd
import matplotlib.pyplot as plt
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score


def modelfit(alg, dtrain, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=y.values)
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print('N_estimators: {}'.format(cvresult.shape[0]))
    
    #Fit the algorithm on the data
    alg.fit(dtrain, y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
        
    #Print model report:
    print("\nModel Report")
    print("R2 train: %.4g" % r2_score(y.values, dtrain_predictions))
    cross = cross_val_score(alg, dtrain, y, scoring = 'r2', n_jobs = -1, cv = 5)
    print(cross)
    print('R2 cross-val: {}'.format(cross.mean()))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
  



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


xgb1 = XGBRegressor(learning_rate =0.05,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb1, train)
# Cross 0.5798
# numero alberi 61

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
utils.grid_search(xgb1, param_test1, train, y)
# max_depth: 3, min_child_weight: 5

param_test2 = {
 'max_depth':range(4,6),
 'min_child_weight':range(2,5)
}
utils.grid_search(xgb1, param_test2, train, y)
# max_depth: 4, min_child_weight: 4

xgb2 = XGBRegressor(learning_rate =0.05,
                     n_estimators=1000,
                     max_depth=4,
                     min_child_weight=4,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb2, train)
# Cross 0.5855
# numero alberi 65

param_test3 = {
 'gamma': [x / 10 for x in range(0, 105, 5)]
}
utils.grid_search(xgb2, param_test3, train, y)
# gamma: 7

param_test4 = {
 'gamma': [x / 10 for x in range(66, 75)]
}
utils.grid_search(xgb2, param_test4, train, y)
# gamma: 6.8


xgb3 = XGBRegressor(learning_rate =0.05,
                     n_estimators=1000,
                     max_depth=4,
                     min_child_weight=4,
                     gamma=6.8,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb3, train)
# Cross 0.5892
# numero alberi 149

param_test5 = {
 'subsample':[i/10.0 for i in range(4,11)],
 'colsample_bytree':[i/10.0 for i in range(4,11)]
}
utils.grid_search(xgb3, param_test5, train, y)
# subsample: 1, colsample: 0.5

param_test6 = {
 'colsample_bytree':[i/100 for i in range(45,56, 5)],
 'subsample':[i/100 for i in range(95, 101, 5)]
}
utils.grid_search(xgb3, param_test6, train, y)
#subsample: 0.95, colsample: 0.45

param_test7 = {
 'colsample_bytree':[i/100 for i in range(25,46, 5)],
 'subsample':[0.95]
}
utils.grid_search(xgb3, param_test7, train, y)
#subsample: 0.95, colsample: 0.3

param_test8 = {
 'colsample_bytree':[i/100 for i in range(5,31, 5)],
 'subsample':[0.95]
}
utils.grid_search(xgb3, param_test8, train, y)
#subsample: 0.95, colsample: 0.3


xgb4 = XGBRegressor(learning_rate =0.05,
                     n_estimators=1000,
                     max_depth=4,
                     min_child_weight=4,
                     gamma=6.8,
                     subsample=0.95,
                     colsample_bytree=0.3,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb4, train)
# Cross 0.5931
# numero alberi 262

param_test9 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
utils.grid_search(xgb4, param_test9, train, y)
# reg_alpha = 0.01

param_test10 = {
 'reg_alpha':[x / 100 for x in range(2, 11, 2)]
}
utils.grid_search(xgb4, param_test10, train, y)
# reg_alpha = 0.02

param_test11 = {
 'reg_alpha':[0.005, 0.01, 0.015, 0.02]
}
utils.grid_search(xgb4, param_test11, train, y)
# reg_alpha = 0.02


xgb5 = XGBRegressor(learning_rate =0.05,
                     n_estimators=1000,
                     max_depth=4,
                     min_child_weight=4,
                     gamma=6.8,
                     subsample=0.95,
                     colsample_bytree=0.3,
                     reg_alpha = 0.02,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb5, train)
# Cross 0.5934
# numero alberi 258

param_test12 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
utils.grid_search(xgb4, param_test12, train, y)
# gamma: 1

param_test13 = {
 'reg_lambda':[x / 10 for x in range(5, 105, 5)]
}
utils.grid_search(xgb4, param_test13, train, y)
# gamma: 6


xgb6 = XGBRegressor(learning_rate =0.05,
                     n_estimators=3000,
                     max_depth=4,
                     min_child_weight=4,
                     gamma=6.8,
                     subsample=0.95,
                     colsample_bytree=0.3,
                     reg_alpha = 0.02,
                     reg_lambda = 6,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb6, train)
# Cross 0.5937
# numero alberi 317

#param_test13 = {
# 'learning_rate':[0.001, 0.005, 0.01, 0.015, 0.02]
#}
#utils.grid_search(xgb6, param_test13, train, y)
# gamma: 0.02


xgb7 = XGBRegressor(learning_rate =0.01,
                     n_estimators=6000,
                     max_depth=4,
                     min_child_weight=4,
                     gamma=6.8,
                     subsample=0.95,
                     colsample_bytree=0.3,
                     reg_alpha = 0.02,
                     reg_lambda = 6,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state = 1024)
modelfit(xgb7, train)
# Cross 0.5937
# numero alberi 1444

cross = cross_val_score(xgb7, train, y, scoring = 'r2', n_jobs = -1, cv = 20)
print(cross)
print('Cross val R2: {}'.format(cross.mean()))

xgb7.fit(train, y)
y_pred = xgb7.predict(test)
# Denormalize the predictions
y_pred_denorm = scaler.inverse_transform(y_pred)
sub = pd.DataFrame(data = {'ID': test_ids, 'y': y_pred_denorm})
sub.to_csv('submissions/xgboost_onehot_full.csv', index = False)