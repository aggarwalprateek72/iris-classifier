# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:36:34 2019

@author: Prateek
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset= pd.read_csv('iris.csv')

#initializing the independent and dependent variables
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:, 4].values

#dealing with the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y= LabelEncoder()
Y= labelencoder_Y.fit_transform(Y)

#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.25, random_state=0)

#Applying the feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

#Making the classifier
from sklearn.svm import SVC
classifier= SVC(random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred= classifier.predict(X_test)

#Verifying the results using confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)



