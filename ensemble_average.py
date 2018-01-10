import pandas as pd
import seaborn as sns

path = 'submissions/'
files = [
        'svr_g0.0001_eps_0.5_c1.6_3feat.csv',
        'lasso_3feat_a0006.csv',
        'xgboost_df_PCA_ICA_max4_notebook.csv',
        'sub.csv',
        'xgboost_onehot_full.csv'
        ]
columns = ['y_{}'.format(i) for i in range(len(files))]
weights = pd.Series([2/6, 1/6, 1/6, 1/6, 1/6], index = columns)

test_ids = pd.read_csv(path + files[0])['ID']
predictions = {'y_{}'.format(i): pd.read_csv(path + files[i])['y'] for i in range(len(files))}
df = pd.DataFrame(predictions)

# corelation between predictions
corr = df.corr()
sns.heatmap(corr)

df['y'] = df[columns].sum(axis = 1) / len(files)
df['y00'] = (df[columns] * weights.transpose()).sum(1)

sub = pd.DataFrame({'ID': test_ids, 'y': df['y00']})
sub.to_csv(path + 'ensemble_weightedvoting_svr_lasso_rf_2xgb.csv', index = False)
