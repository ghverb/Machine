# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:06:05 2020

@author: gwenv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_diabetes
dia_data = load_diabetes()
dia_data = pd.DataFrame(data = np.c_[dia_data['data'],dia_data['target']],columns=list(dia_data['feature_names']) + ['target'])


'''STEP ONE: Splitting the data into training and testing sets'''
#Create An array only containing X values
X = dia_data['bmi'].values
X = X.reshape(-1,1)

#Create an vector only containing y values
y = dia_data['target'].values
y = y.reshape(-1,1)

#Split data into X-training set, X-testing set, y-training set, y-testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3,random_state = 1)

'''STEP TWO: Building the model'''
#Create Model Object
linmodel = LinearRegression()

#Train your model object onto your training sets
linmodel.fit(X_train,y_train)
print(linmodel.coef_)

'''STEP THREE: Model Testing and DIAGNOSTICS'''
#Pass Your training set through your model to see predictions on data it was trained on
predictions_train = linmodel.predict(X_train)

#Pass your testing set through your model to see predictions on new data
predictions_test = linmodel.predict(X_test)

#Calculate performance metrics:
print('Simple Lin Model Training R2:',r2_score(y_train,predictions_train))
print('Simple Lin Model Testing R2:',r2_score(y_test,predictions_test))
print('Simple Lin Model Training MSE:', mean_squared_error(y_train,predictions_train))
print('Simple Lin Model Testing MSE', mean_squared_error(y_test,predictions_test))

'''Now lets build a multiple linear model'''

'''STEP ONE: Splitting the data into training and testing sets'''
#Create An array only containing X values
y = dia_data['target'].values
dia_data.drop(columns=['sex','target'] , inplace=True)
X = dia_data.values

#Split data into X-training set, X-testing set, y-training set, y-testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3,random_state = 1)

'''STEP TWO: Building the model'''
#Create Model Object
multlinmodel = LinearRegression()

#Train your model object onto your training sets
multlinmodel.fit(X_train,y_train)
'''STEP THREE: Model Testing and DIAGNOSTICS'''
#Pass Your training set through your model to see predictions on data it was trained on
predictions_train = multlinmodel.predict(X_train)

#Pass your testing set through your model to see predictions on new data
predictions_test = multlinmodel.predict(X_test)
#Calculate Performance Metrics:

print("Multi Lin Training R2:",r2_score(y_train,predictions_train))
print('Multi Lin Testing R2:',r2_score(y_test,predictions_test))
print('Multi Lin MSE - Training:',mean_squared_error(y_train,predictions_train))
print('Multi Lin MSE - Testing:',mean_squared_error(y_test,predictions_test))

'''Now lets fit a Complex Polynomial to our data'''
'''STEP ONE: Splitting the data into training and testing sets'''
#We can just use the X_train, y_train, X_test, y_test from before

'''STEP TWO: Building the model'''
#Create Model Object , pipeline, allows you to put in multiple functions
polyreg = make_pipeline(PolynomialFeatures(degree= 4), LinearRegression())

#Train your model object onto your training sets
polyreg.fit(X_train,y_train)

'''STEP THREE: Model Testing and DIAGNOSTICS'''
#Pass Your training set through your model to see predictions on data it was trained on
predictions_train = polyreg.predict(X_train)
#Pass your testing set through your model to see predictions on new data
predictions_test = polyreg.predict(X_test)

print('Polynomial R2 Training:',r2_score(y_train,predictions_train))
print('Polynomial R2 Testing:',r2_score(y_test,predictions_test))
print('Polynomial MSE Training:',mean_squared_error(y_train,predictions_train))
print('Polynomial MSE Testing:',mean_squared_error,predictions_test)