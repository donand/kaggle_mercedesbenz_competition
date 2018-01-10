import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import utils
from xgboost.sklearn import XGBRegressor
from sklearn.decomposition import FastICA

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
    cross = cross_val_score(alg, dtrain, y, scoring = 'r2', n_jobs = -1, cv = 4)
    print('R2 cross-val: {}'.format(cross.mean()))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train['y']
test_ids = test['ID']
train, test = utils.label_encode_categorical(train, test)
train = train.drop(['ID', 'y'], axis = 1)
test = test.drop('ID', axis = 1)

### PCA ###
df_pca, df_test_pca = utils.pca(train, test, 0.99)

### ICA ###
columns = ['ICA_{}'.format(i) for i in range(10)]
ica = FastICA(n_components=10, random_state = 42)
df_ica = pd.DataFrame(ica.fit_transform(train), columns = columns)
df_test_ica = pd.DataFrame(ica.transform(test), columns = columns)

train = pd.concat([train, df_pca, df_ica], axis = 1)
test = pd.concat([test, df_test_pca, df_test_ica], axis = 1)


predictors = [x for x in train.columns if x not in ['ID', 'y']]
xgb1 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb1, train, predictors)
# Cross 0.5541
# numero alberi 53

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test1, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch1.fit(train[predictors],y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# max_depth: 3, min_child_weight: 5

param_test2 = {
 'max_depth': [1,2,3,4],
 'min_child_weight':range(4,11,1)
}
gsearch2 = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test2, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch2.fit(train[predictors],y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
# max_depth: 2, min_child_weight: 9

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

xgb1.set_params(min_child_weight = 28)
xgb1.set_params(max_depth = 3)
param_test3 = {
 'gamma': [x / 10 for x in range(0, 31)]
}
gsearch3 = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test3, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch3.fit(train[predictors],y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# gamma = 0
# max_depth = 3, min_child_weight = 28, gamma = 0
xgb2 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=28,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb2, train, predictors)
# Cross 0.5542
# numero alberi 60


param_test4 = {
 'subsample':[i/10.0 for i in range(4,11)],
 'colsample_bytree':[i/10.0 for i in range(4,11)]
}
gsearch4 = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test4, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch4.fit(train[predictors],y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
# {'colsample_bytree': 0.8, 'subsample': 0.8}

param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test5, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch5.fit(train[predictors],y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
# 'colsample_bytree': 0.8, 'subsample': 0.8

xgb2.set_params(colsample_bytree = 0.8, subsample = 0.8)
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test6, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch6.fit(train[predictors],y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
# reg_alpha = 0.01

param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test7, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch7.fit(train[predictors],y)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
# reg_alpha = 0.01

param_test7b = {
 'reg_alpha': [x / 1000 for x in range(6, 30, 2)]
}
gsearch7b = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test7b, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch7b.fit(train[predictors],y)
gsearch7b.grid_scores_, gsearch7b.best_params_, gsearch7b.best_score_
# reg_alpha = 0.026

### XGB 3 ###

# max_depth = 3, min_child_weight = 28, gamma = 0
# colsample_bytree: 0.8, subsample: 0.8, reg_alpha: 0.026
xgb3 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=28,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha = 0.026,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb3, train, predictors)
# Cross 0.5542
# numero alberi 60

### DA FARE reg_lambda
param_test8 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch8 = GridSearchCV(estimator = xgb3, 
                        param_grid = param_test8, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch8.fit(train[predictors],y)
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_
# reg_lambda = 1

param_test9 = {
 'reg_lambda':[x / 10 for x in range(5, 105, 5)]
}
gsearch9 = GridSearchCV(estimator = xgb3, 
                        param_grid = param_test9, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch9.fit(train[predictors],y)
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_
# reg_lambda = 3

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

# Decrease learning rate
xgb4 = XGBRegressor(learning_rate =0.01,
                     n_estimators=5000,
                     max_depth=3,
                     min_child_weight=28,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha = 0.026,
                     reg_lambda = 16,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb4, train, predictors)
# Cross 0.5623
# numero alberi 648

cross = cross_val_score(xgb4, train, y, scoring = 'r2', n_jobs = -1, cv = 10)
print('Cross val R2: {}'.format(cross.mean()))

utils.generate_submission(xgb4, train, y,
                          test, test_ids,
                          'xgboost_PCA_ICA_tuned.csv')
                          