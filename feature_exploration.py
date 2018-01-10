import pandas as pd
import utils
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Analisi categorical features
categorical_columns = ['X' + str(i) for i in range(0,9) if i != 7]
#df, df_test = utils.label_encode_categorical(train, test)
sns.stripplot(x = 'y', y = 'X6', data = train, jitter = True)

# Count unique car combinations and their y mean
group = train[categorical_columns + ['y']].groupby(categorical_columns)
mean_per_car = group.mean()
print('Unique car combinations over all the samples: {}/{}'.format(len(group), len(train)))

# Count the duplicates in car features
car_dup = group.count()
n_car_dup = car_dup[car_dup.y > 1]
print('There are {} car combinations that are repeated 2 or more times'.format(len(n_car_dup)))

# Plot X0
x0 = train[['X0', 'y']].groupby('X0', as_index = False).mean()
x0.columns = ['X0', 'X0_mean']
x0 = x0.sort_values('X0_mean').drop('X0_mean', axis = 1)
y = train.y
sns.boxplot(x = 'X0', y = 'y', data = train, order = x0.X0)

# Outlier analysis
out = train[train.y > 250]
n_zero = np.count_nonzero(out.values)