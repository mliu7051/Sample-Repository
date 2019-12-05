# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import StandardScaler, Normalizer


literature_data = pd.read_csv('Literature Data.csv')
# dropped add3, should add back when there are more third additives
literature_data = literature_data.drop(columns=['add3wt', 'add2wt', 'Article', '1author', 'peowt', 'polymw', 'add1', 'add2', 'add3'])
literature_data = literature_data.fillna(0)

litfeat = literature_data.drop(columns=['conduct'])
litprop = literature_data['conduct']

X_train, X_test, y_train, y_test = train_test_split(litfeat, litprop, test_size=0.2, random_state=39)
randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)

predictedy = randomforestmodel.predict(X_test)
score = randomforestmodel.score(X_test, y_test)
feat = randomforestmodel.feature_importances_


"""
normalizer = Normalizer().fit(litfeat)
scaler = StandardScaler().fit(litfeat)

X_train = normalizer.transform(X_train)
#X_train = scaler.transform(X_train)
X_test = normalizer.transform(X_test)
#X_test = scaler.transform(X_test)



linreg = lr(normalize=True)
linreg.fit(X_train, y_train)
predictedy = linreg.predict(X_test)
coef = linreg.coef_
intercept = linreg.intercept_
"""
















"""
# import csv file later

# dropping PEO, salt, thickness, and typesalt columns
electrolyte_data = pd.read_csv('Electrolyte Data.csv')
electrolyte_data = electrolyte_data.drop(columns=['PEO','salt','thickness','typesalt'])

# filling in empty columns with random numbers
# electrolyte_data['typesalt'] = 'LiFSI'

rand_conduct = np.random.randint(-10,-3, size=26)
rand_conduct = rand_conduct.astype(float)
electrolyte_data['conduct'] = 10**rand_conduct

rand_iontransfer = np.random.randint(50,101, size=26)
electrolyte_data['iontransfer'] = rand_iontransfer/100

rand_Vmax = np.random.randint(3,5, size=26)
electrolyte_data['Vmax'] = rand_Vmax

rand_Vmin = np.random.randint(1,3, size=26)
electrolyte_data['Vmin'] = rand_Vmin

rand_decomptemp = np.random.randint(200,400, size=26)
electrolyte_data['decomptemp'] = rand_decomptemp


# drops empty row at bottom
electrolyte_data = electrolyte_data.dropna()


# split data into two dataframes
electrolyte_features = electrolyte_data[['saltpc','temp']]
electrolyte_properties = electrolyte_data[['conduct','iontransfer','Vmax','Vmin','decomptemp']]

# creating a model
X_train, X_test, y_train, y_test = train_test_split(electrolyte_features, electrolyte_properties, test_size=0.2, random_state=64)
randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)

predictedy = randomforestmodel.predict(X_test)
score = randomforestmodel.score(X_test, y_test)

residuals = y_test-predictedy
sns.distplot(residuals['conduct'], bins=10)
sns.plt.show()
"""