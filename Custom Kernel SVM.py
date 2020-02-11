# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:31:22 2020

@author: xf290
"""

import numpy as np
import pandas as pd
from sklearn import *


mldata = pd.read_csv('all data.csv')
mldata = mldata.drop(columns=['1author', 'LiTFSIwt2', 'temp2', 'exponent2', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'])
#mldata = mldata.iloc[118:138]

"""
def my_kernel(X, Y):
    

    
"""

litfeat = mldata.drop(columns = ['exponent'])
litprop = mldata['exponent']
X_train, X_test, y_train, y_test = model_selection.train_test_split(litfeat, litprop, test_size=0.2, random_state=23)

svr = svm.SVR()
svr.fit(X_train, y_train)

predicted = svr.predict(X_test)