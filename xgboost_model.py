import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import utils
from sklearn.metrics import r2_score
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FastICA

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

y = df['y']
df = df.drop(['ID', 'y'], axis = 1)
test_ids = df_test['ID']
df_test = df_test.drop('ID', axis = 1)
df, df_test = utils.label_encode_categorical(df, df_test)


### PCA ###
df_pca, df_test_pca = utils.pca(df, df_test, 10)

### ICA ###
columns = ['ICA_{}'.format(i) for i in range(10)]
ica = FastICA(n_components=10, random_state = 42)
df_ica = pd.DataFrame(ica.fit_transform(df), columns = columns)
df_test_ica = pd.DataFrame(ica.transform(df_test), columns = columns)

### XGBOOST ###
y_mean = y.mean()
# prepare dict of params for xgboost to run with
xgb_params = { 
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'colsample_bytree' : 0.7,
    'lambda': 2
}


### NO PCA ###
# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(df, y)
dtest = xgb.DMatrix(df_test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=2000, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
                   feval=xgb_r2_score
                  )

# Plot R2 and RMSE on training and validation sets
'''
plt.plot(cv_result.index, cv_result['train-r2-mean'], c = 'blue')
plt.plot(cv_result.index, cv_result['test-r2-mean'], c = 'red')
plt.show()
plt.plot(cv_result.index, cv_result['train-rmse-mean'], c = 'blue')
plt.plot(cv_result.index, cv_result['test-rmse-mean'], c = 'red')
plt.show()
'''
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
xgb_params['silent'] = 0
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
print('Train R2: {}'.format(r2_score(dtrain.get_label(), model.predict(dtrain))))

m = XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=num_boost_rounds,
                 n_jobs=-1, subsample=0.95, silent = False)
cross = cross_val_score(m, df, y, scoring = 'r2', cv = 5, n_jobs = -1, verbose = True)
cross.mean()

# Generate submission
pred = model.predict(dtest)
sub = pd.DataFrame(data = {'ID': test_ids, 'y': pred})
sub.to_csv('data/xgboost_first_try.csv', index = False)



### PCA + ICA ###
# form DMatrices for Xgboost training
df_red = pd.concat([df, df_pca, df_ica], axis = 1)
df_test_red = pd.concat([df, df_test_pca, df_test_ica], axis = 1)
dtrain_red = xgb.DMatrix(df_red, y)
dtest_red = xgb.DMatrix(df_test_red)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain_red, 
                   num_boost_round=5000, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
                   feval = xgb_r2_score
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)
model = xgb.train(xgb_params, dtrain_red, num_boost_round=num_boost_rounds)

m = XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=num_boost_rounds,
                 n_jobs=-1, subsample=0.9, silent = False, colsample_bylevel = 0.7,
                 reg_lambda=2)
cross = cross_val_score(m, df_red, y, scoring = 'r2', cv = 5, n_jobs = -1, verbose = True)
cross.mean()

# Generate submission
pred = model.predict(dtest_red)
sub = pd.DataFrame(data = {'ID': test_ids, 'y': pred})
sub.to_csv('data/xgboost_df_PCA_ICA_max4_eta0.05_sub0.9_col0.7.csv', index = False)