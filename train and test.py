# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt


literature_data = pd.read_csv('Feed In Data.csv')
#Dataset 2
#literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 3
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])

#Dataset 1
#literature_data = literature_data.drop(columns=['1author3', '1author2', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])
literature_data = literature_data.dropna()

litfeat = literature_data.drop(columns = ['exponent'])
litprop = literature_data['exponent']
X_train, X_test, y_train, y_test = train_test_split(litfeat, litprop, test_size=0.2, random_state=4)

"""
#----------------------------linear regression train and test------------------------------------------------------------------

linreg = lr(normalize=True)
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)
#coef = linreg.coef_
#intercept = linreg.intercept_
#score = linreg.score(X_test, y_test)
linreg_rmse = mean_squared_error(y_test, linreg_pred)
print('linreg MAE: ' + str(sum(abs(linreg_pred - y_test))/(len(y_test))))
print('linreg RMSE: ' + str(np.sqrt(linreg_rmse)))

#-------------------------------decision tree train and test------------------------------------------------------------------
dectree = dtr()
dectree.fit(X_train,y_train)

dectree_pred = dectree.predict(X_test)
#score = dectree.score(X_test, y_test)
dectree_rmse = mean_squared_error(y_test, dectree_pred)
print('dectree MAE: ' + str(sum(abs(dectree_pred - y_test))/(len(y_test))))
print('dectree RMSE: ' + str(np.sqrt(dectree_rmse)))
"""
#-------------------------------random forest train and test---------------------------------------------------------------------
randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)

rf_pred = randomforestmodel.predict(X_test)
#score = randomforestmodel.score(X_test, y_test)
rf_rmse = mean_squared_error(y_test, rf_pred)
print('rf MAE: ' + str(sum(abs(rf_pred - y_test))/(len(y_test))))
print('rf RMSE: ' + str(np.sqrt(rf_rmse)))

print('rf percent error: ' + str(sum(((abs(rf_pred - y_test)/y_test)*100))/(len(y_test))))
"""
residuals = y_test-rf_pred
sns.distplot(residuals, bins=30)
plt.show()

importances = randomforestmodel.feature_importances_


#------------------------------support vector machine 1 train and test------------------------------------------------------------

svr = svm.SVR()
svr = svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
svr_rmse = mean_squared_error(y_test, svr_pred)

print('svr MAE: ' + str(sum(abs(svr_pred - y_test))/(len(y_test))))
print('svr RMSE: ' + str(np.sqrt(svr_rmse)))


#-------------------------------support vector machine 2 train and test----------------------

def arrhenius_kernel(X, Y):
    #return np.dot(np.exp((-1)/(X)), Y.T)
    return np.dot(X, Y.T)

svr2 = svm.SVR(kernel = arrhenius_kernel)
svr2 = svr2.fit(X_train, y_train)
svr2_pred = svr2.predict(X_test)
svr2_rmse = mean_squared_error(y_test, svr2_pred)

print('svr2 MAE: ' + str(sum(abs(svr2_pred - y_test))/(len(y_test))))
print('svr2 RMSE: ' + str(np.sqrt(svr2_rmse)))


#-------------------------------support vector machine 3 train and test-----------------------

def importances_kernel(X_1, X_2):
    M = np.array([[importances[0], 0], [0, importances[1]]])
    return np.dot(np.dot(X_1, M), X_2.T)

svr3 = svm.SVR(kernel = importances_kernel)
svr3 = svr3.fit(X_train, y_train)
svr3_pred = svr3.predict(X_test)
svr3_rmse = mean_squared_error(y_test, svr3_pred)

print('svr3 MAE: ' + str(sum(abs(svr3_pred - y_test))/(len(y_test))))
print('svr3 RMSE: ' + str(np.sqrt(svr3_rmse)))
"""