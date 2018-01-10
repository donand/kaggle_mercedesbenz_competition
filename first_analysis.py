import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import utils

# MODELS #
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


### BEST MODEL: RF n = 1000, max_depth = 5, no PCA


df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

y = df['y']
df = df.drop(['ID', 'y'], axis = 1)
df, df_test = utils.label_encode_categorical(df, df_test)

test_ids = df_test['ID']
df_test = df_test.drop('ID', axis = 1)

### PROVA PCA ###
df_pca, df_test_pca = utils.pca(df, df_test, 0.99)
plt.scatter(df_pca['PCA_0'], df_pca['PCA_1'], s = 1)
plt.xlabel('PCA_0')
plt.ylabel('PCA_1')
plt.show()
plt.scatter(df_pca['PCA_0'], y, s = 1)
plt.xlabel('PCA_0')
plt.ylabel('y')
plt.show()
plt.scatter(df_pca['PCA_1'], y, s = 1)
plt.xlabel('PCA_1')
plt.ylabel('y')
plt.show()

### PROVA RANDOM FOREST ###
rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, max_depth=3)
cross = cross_val_score(rf, df, y, verbose=True, cv = 5, scoring='r2', n_jobs = -1)
cross.mean()


# NO PCA, n = 500  --> best_depth = 3
# NO PCA, n = 1500 --> best_depth = 3
depths = range(1, 11)
models = [RandomForestRegressor(n_estimators=1500, n_jobs=-1, max_depth=i) for i in depths]
values, scores, best_v, best_s = utils.parameter_search(df, y, models, depths, 'r2', xlabel = 'Max depth')

n = range(50, 1001, 50)
models = [RandomForestRegressor(n_estimators=i, n_jobs=-1) for i in n]
values, scores, best_v, best_s = utils.parameter_search(df, y, models, n, 'r2', xlabel = '# alberi')


### K-NN ###
knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
cross_knn = cross_val_score(knn, df, y, verbose = True, n_jobs=-1, scoring = 'r2', cv = 10)
cross_knn.mean()

# NO PCA, K = 10, R2 = 0.328
n = range(1, 50)
models = [KNeighborsRegressor(n_neighbors=i, n_jobs=-1) for i in n]
v_knn, s_knn, best_v_knn, best_s_knn = utils.parameter_search(df, y, models, n, 'r2', 'K')
print(best_v_knn, best_s_knn)

# CON PCA, K = 14, R2 = 0.325
n = range(1, 50)
models = [KNeighborsRegressor(n_neighbors=i, n_jobs=-1) for i in n]
v_knn, s_knn, best_v_knn, best_s_knn = utils.parameter_search(df_pca, y, models, n, 'r2', 'K')
print(best_v_knn, best_s_knn)


### GENERATE SUBMISSION ###
rf = RandomForestRegressor(n_estimators=1000, verbose=True, n_jobs=-1, max_depth=5)
utils.generate_submission(rf, df, y, df_test, test_ids, 'rf_n1000_d5.csv')
