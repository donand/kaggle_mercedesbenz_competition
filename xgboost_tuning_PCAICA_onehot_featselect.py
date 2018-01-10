import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import utils
from xgboost.sklearn import XGBRegressor
from sklearn.decomposition import FastICA
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y.values)
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print('N_estimators: {}'.format(cvresult.shape[0]))
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
        
    #Print model report:
    print("\nModel Report")
    print("R2 train: %.4g" % r2_score(y.values, dtrain_predictions))
    cross = cross_val_score(alg, dtrain, y, scoring = 'r2', n_jobs = -1, cv = 5)
    print(cross)
    print('R2 cross-val: {}'.format(cross.mean()))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
  
    
def grid_search(estimator, params):
    gsearch1 = GridSearchCV(estimator = estimator, 
                        param_grid = params, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
    gsearch1.fit(train[predictors],y)
    return gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Set column type to categorical
categorical_columns = ['X' + str(i) for i in range(0,9) if i != 7]
for c in categorical_columns:
    categories = list(set(train[c].unique().tolist() + test[c].unique().tolist()))
    train[c] = train[c].astype('category', categories = categories)
    test[c] = test[c].astype('category', categories = categories)

# Eliminare una colonna per ogni colonna categorical
train = pd.get_dummies(train)
test = pd.get_dummies(test)

y = train['y']
test_ids = test['ID']
#train, test = utils.label_encode_categorical(train, test)
train = train.drop(['ID', 'y'], axis = 1)
test = test.drop('ID', axis = 1)


### PCA ###
df_pca, df_test_pca = utils.pca(train, test, 20)

### ICA ###
columns = ['ICA_{}'.format(i) for i in range(10)]
ica = FastICA(n_components=10, random_state = 42, max_iter = 10000, tol = 0.0005)
df_ica = pd.DataFrame(ica.fit_transform(train), columns = columns)
df_test_ica = pd.DataFrame(ica.transform(test), columns = columns)

train = pd.concat([train, df_pca, df_ica], axis = 1)
test = pd.concat([test, df_test_pca, df_test_ica], axis = 1)


### FEATURE SELECTION ###
train_orig = train
test_orig = test
sel = SelectKBest(score_func = mutual_info_regression, k = 42)
train = pd.DataFrame(sel.fit_transform(train, y))
scores = pd.Series(sel.scores_)
pvalues = sel.pvalues_
test = pd.DataFrame(sel.transform(test))
len(scores[scores >= 0.1])

predictors = [x for x in train.columns if x not in ['ID', 'y']]
xgb1 = XGBRegressor(learning_rate =0.1,
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
                     random_state=27)
modelfit(xgb1, train, predictors)
# Cross 0.5428
# numero alberi 27

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
grid_search(xgb1, param_test1)
# max_depth: 3, min_child_weight: 5

param_test2 = {
 'max_depth': [1,2,3,4],
 'min_child_weight':range(4,10)
}
grid_search(xgb1, param_test2)
# max_depth: 3, min_child_weight: 4

param_test2b = {
 'min_child_weight':range(19, 31)
}
xgb1.set_params(max_depth = 2)
gsearch2b = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test2b, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch2b.fit(train[predictors],y)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_
# max_depth = 2, min_child_weight = 27

param_test2c = {
 'max_depth': [2,3,4,5],
 'min_child_weight':range(4,31)
}
gsearch2c = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test2c, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch2c.fit(train[predictors],y)
gsearch2c.grid_scores_, gsearch2c.best_params_, gsearch2c.best_score_
# max_depth: 3, min_child_weight: 28


# max_depth = 3, min_child_weight = 4
xgb2 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     base_score=y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb2, train, predictors)
# Cross 0.5517
# numero alberi 31

param_test3 = {
 'gamma': [x / 10 for x in range(0, 105, 5)]
}
grid_search(xgb2, param_test3)
# gamma = 4

# max_depth = 3, min_child_weight = 4, gamma = 4
xgb3 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=4,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb3, train, predictors)
# Cross 0.5614
# numero alberi 31


param_test4 = {
 'subsample':[i/10.0 for i in range(4,11)],
 'colsample_bytree':[i/10.0 for i in range(4,11)]
}
grid_search(xgb3, param_test4)
# {'colsample_bytree': 1, 'subsample': 1}

param_test5 = {
 'subsample':[i/100.0 for i in range(95,105,5)],
 'colsample_bytree':[i/100.0 for i in range(95,105,5)]
}
grid_search(xgb3, param_test5)
# 'colsample_bytree': 0.95, 'subsample': 1

# max_depth = 3, min_child_weight = 4, gamma = 4
# colsample = 0.95, subsample = 1
xgb4 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=4,
                     subsample=1,
                     colsample_bytree=0.95,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb4, train, predictors)
# Cross 0.5653
# numero alberi 29


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
grid_search(xgb4, param_test6)
# reg_alpha = 0.01

param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
grid_search(xgb4, param_test7)
# reg_alpha = 0.05

param_test7b = {
 'reg_alpha': [x / 100 for x in range(2, 30, 2)]
}
grid_search(xgb4, param_test7b)
# reg_alpha = 0.06

param_test7c = {
 'reg_alpha': [x / 100 for x in range(5, 8)]
}
grid_search(xgb4, param_test7c)
# reg_alpha = 0.05



# max_depth = 3, min_child_weight = 4, gamma = 4
# colsample_bytree: 0.95, subsample: 1, reg_alpha: 0.05
xgb5 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=4,
                     subsample=1,
                     colsample_bytree=0.95,
                     reg_alpha = 0.05,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb5, train, predictors)
# Cross 0.5651
# numero alberi 32

### reg_lambda
param_test8 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
grid_search(xgb5, param_test8)
# reg_lambda = 1

param_test9 = {
 'reg_lambda':[x / 10 for x in range(5, 105, 5)]
}
grid_search(xgb5, param_test9)
# reg_lambda = 6.5

param_test10 = {
 'reg_lambda':[x / 10 for x in range(100, 305, 5)]
}
gsearch10 = GridSearchCV(estimator = xgb3, 
                        param_grid = param_test10, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch10.fit(train[predictors],y)
gsearch10.grid_scores_, gsearch10.best_params_, gsearch10.best_score_
# reg_lambda = 16

xgb6 = XGBRegressor(learning_rate =0.1,
                     n_estimators=5000,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=4,
                     subsample=1,
                     colsample_bytree=0.95,
                     reg_alpha = 0.05,
                     reg_lambda = 6.5,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb6, train, predictors)
# Cross 0.5663
# N alberi 36

param_test11 = {
 'scale_pos_weight':[1e-5, 1e-2, 0.1, 1, 100, 1000]
}
grid_search(xgb6, param_test11)
# scale_pos_weight = 1 USELESS

param_test12 = {
 'max_delta_step':[0, 1e-5, 1e-2, 0.1, 1, 100, 1000]
}
grid_search(xgb6, param_test12)
# max_delta_step = 0 USELESS



# Decrease learning rate
xgb7 = XGBRegressor(learning_rate =0.01,
                     n_estimators=5000,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=4,
                     subsample=1,
                     colsample_bytree=0.95,
                     reg_alpha = 0.05,
                     reg_lambda = 6.5,
                     base_score = y.mean(),
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb7, train, predictors)
# Cross 0.5660
# numero alberi 349

cross = cross_val_score(xgb7, train, y, scoring = 'r2', n_jobs = -1, cv = 10)
print(cross)
print('Cross val R2: {}'.format(cross.mean()))

utils.generate_submission(xgb7, train, y,
                          test, test_ids,
                          'xgboost_PCA_ICA_onehot_featselect.csv')
                          