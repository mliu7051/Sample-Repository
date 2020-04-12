# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:10:26 2020

@author: xf290
"""

import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error
from sklearn import svm


literature_data = pd.read_csv('Feed In Data.csv')
testvalues = pd.read_csv('predicted results.csv')

#Dataset 1
#literature_data = literature_data.drop(columns=['1author3', '1author2', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 2
#literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 3
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])
literature_data = literature_data.dropna()
X_train = literature_data.drop(columns = ['exponent'])
y_train = literature_data['exponent']

"""
#----------------------Linear Regression----------------------------------------------------------------

linreg = lr(normalize=True)
linreg.fit(X_train, y_train)

linreg_pred = linreg.predict(testvalues)

#------------------------------Decision Tree-------------------------------------------------------------
dectree = dtr()
dectree.fit(X_train,y_train)

dectree_pred = dectree.predict(testvalues)
"""
#------------------------------Random Forest------------------------------------------------------------
randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)

rf_pred = randomforestmodel.predict(testvalues)

"""
#------------------------------support vector machine 1 train and test------------------------------------------------------------

svr1 = svm.SVR()
svr1 = svr1.fit(X_train, y_train)
svr1_pred = svr1.predict(testvalues)


#-------------------------------support vector machine 2 train and test----------------------

def arrhenius_kernel(X, Y):
    #return np.dot(np.exp((-1)/(X)), Y.T)
    return np.dot(X, Y.T)

svr2 = svm.SVR(kernel = arrhenius_kernel)
svr2 = svr2.fit(X_train, y_train)
svr2_pred = svr2.predict(testvalues)


#-------------------------------support vector machine 3 train and test-----------------------

def importances_kernel(X_1, X_2):
    M = np.array([[importances[0], 0], [0, importances[1]]])
    return np.dot(np.dot(X_1, M), X_2.T)

svr3 = svm.SVR(kernel = importances_kernel)
svr3 = svr3.fit(X_train, y_train)
svr3_pred = svr3.predict(X_test)






#-----------------------------------------------------------------------------------------------
#----------------------------Model with Fake Data----------------------------------------------------------------------------------


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