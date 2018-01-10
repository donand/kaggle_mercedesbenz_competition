import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import utils
from xgboost.sklearn import XGBRegressor

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['y'].values)
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print('N_estimators: {}'.format(cvresult.shape[0]))
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['y'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
        
    #Print model report:
    print("\nModel Report")
    print("R2 : %.4g" % r2_score(dtrain['y'].values, dtrain_predictions))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train['y']
test_ids = test['ID']
train, test = utils.label_encode_categorical(train, test)


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
gsearch1.fit(train[predictors],train['y'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {
 'max_depth': [1,2,3,4],
 'min_child_weight':range(4,12,1)
}
gsearch2 = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test2, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch2.fit(train[predictors],train['y'])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test2b = {
 'min_child_weight':range(11, 30)
}
xgb1.set_params(max_depth = 3)
gsearch2b = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test2b, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch2b.fit(train[predictors],train['y'])
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

# max_depth = 3, min_child_weight = 17

xgb1.set_params(min_child_weight = 17)
param_test3 = {
 'gamma': [x / 10 for x in range(11, 30)]
}
gsearch3 = GridSearchCV(estimator = xgb1, 
                        param_grid = param_test3, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch3.fit(train[predictors],train['y'])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# gamma = 1.9
# max_depth = 3, min_child_weight = 17, gamma = 1.9
xgb2 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=17,
                     gamma=1.9,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb2, train, predictors)



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
gsearch4.fit(train[predictors],train['y'])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
# {'colsample_bytree': 1.0, 'subsample': 0.7}

param_test5 = {
 'subsample':[i/100.0 for i in range(65,80,5)],
 'colsample_bytree':[i/100.0 for i in range(95,105,5)]
}
gsearch5 = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test5, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch5.fit(train[predictors],train['y'])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
# 'colsample_bytree': 0.95, 'subsample': 0.7

xgb2.set_params(colsample_bytree = 0.95, subsample = 0.7)
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = xgb2, 
                        param_grid = param_test6, 
                        scoring='r2',
                        n_jobs=-1,
                        iid=False, 
                        cv=5)
gsearch6.fit(train[predictors],train['y'])
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
gsearch7.fit(train[predictors],train['y'])
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
gsearch7b.fit(train[predictors],train['y'])
gsearch7b.grid_scores_, gsearch7b.best_params_, gsearch7b.best_score_
# reg_alpha = 0.016

### XGB 3 ###

# max_depth = 3, min_child_weight = 17, gamma = 1.9
# colsample_bytree: 0.95, subsample: 0.7, reg_alpha: 0.016
xgb3 = XGBRegressor(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=3,
                     min_child_weight=17,
                     gamma=1.9,
                     subsample=0.7,
                     colsample_bytree=0.95,
                     reg_alpha = 0.016,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb3, train, predictors)


xgb4 = XGBRegressor(learning_rate =0.01,
                     n_estimators=5000,
                     max_depth=3,
                     min_child_weight=17,
                     gamma=1.9,
                     subsample=0.7,
                     colsample_bytree=0.95,
                     reg_alpha = 0.016,
                     objective= 'reg:linear',
                     n_jobs=-1,
                     scale_pos_weight=1,
                     random_state=27)
modelfit(xgb4, train, predictors)

cross = cross_val_score(xgb4, train.drop(['ID', 'y'], axis = 1), 
                        y, 
                        scoring = 'r2', 
                        n_jobs = -1, 
                        cv = 10)
cross.mean()

utils.generate_submission(xgb4, train.drop(['ID', 'y'], axis = 1), y,
                          test.drop('ID', axis = 1), test_ids,
                          'xgboost_original_ds_tuned.csv')
                          