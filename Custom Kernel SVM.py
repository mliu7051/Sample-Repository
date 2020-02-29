# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:31:22 2020

@author: xf290
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
        

literature_data = pd.read_csv('Feed In Data.csv')
literature_data = literature_data.drop(columns=['1author'])
literature_data = literature_data.dropna()

X = literature_data.drop(columns = ['exponent'])
#X = literature_data.drop(columns = ['exponent', 'LiTFSIwt'])
y = literature_data['exponent']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

# Random Forest Regression
randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)
rf_pred = randomforestmodel.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred)

# Calculate MAE
print('rf MAE: ' + str(sum(abs(rf_pred - y_test))/(len(y_test))))
print('rf RMSE: ' + str(np.sqrt(rf_rmse)))


def arrhenius_kernel(X, Y):
    #return np.dot(np.exp((-1)/(X)), Y.T)
    return np.dot(X, Y.T)

# Support Vector Regression
svr = svm.SVR(kernel = arrhenius_kernel)
svr = svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
svr_rmse = mean_squared_error(y_test, svr_pred)

# Calculate MAE
print('svr MAE: ' + str(sum(abs(svr_pred - y_test))/(len(y_test))))
print('svr RMSE: ' + str(np.sqrt(svr_rmse)))

comparison_frame = pd.DataFrame()
comparison_frame['y_test'] = y_test
comparison_frame['rf_pred'] = rf_pred
comparison_frame['svr_pred'] = svr_pred





















"""
import numpy as np
import pandas as pd
import sympy as sp
from sklearn import *
from sklearn.ensemble import RandomForestRegressor as rfr


mldata = pd.read_csv('all data.csv')
mldata = mldata.drop(columns=['1author', 'LiTFSIwt2', 'temp2', 'exponent2', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'temp C'])

litfeat = mldata.drop(columns = ['exponent'])
litprop = mldata['exponent']
X_train, X_test, y_train, y_test = model_selection.train_test_split(litfeat, litprop, test_size=0.2, random_state=23)


randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)
rf_pred = randomforestmodel.predict(X_test)
print(sum(abs(rf_pred - y_test))/(len(y_test)))

#--------------------------------------------------------------------------------------------------------

def arrhenius_kernel(exponent, tempK):
    tempK = sp.Symbol('x')
    deriv = sp.diff(exponent/(1000/tempK), tempK)
    Ea = deriv * -8.314
    return Ea

svr = svm.SVR(kernel = arrhenius_kernel)
svr = svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)


print(sum(abs(svr_pred - y_test))/(len(y_test)))

comparison_frame = pd.DataFrame()
comparison_frame['y_test'] = y_test
comparison_frame['rf_pred'] = rf_pred
comparison_frame['svr_pred'] = svr_pred
"""
