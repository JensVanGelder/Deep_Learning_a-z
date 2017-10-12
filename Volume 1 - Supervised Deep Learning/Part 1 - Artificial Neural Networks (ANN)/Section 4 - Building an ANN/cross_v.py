# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:39:26 2017

@author: jens
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu'))
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

def cr_val():
    dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    Y = dataset.iloc[:, 13].values
    label_encoder_X1 = LabelEncoder()
    label_encoder_X2 = LabelEncoder()
    X[:, 1] = label_encoder_X1.fit_transform(X[:, 1])
    X[:, 2] = label_encoder_X2.fit_transform(X[:, 2])
    ohe = OneHotEncoder(categorical_features=[1])
    X = ohe.fit_transform(X).toarray()    
    X = X[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    ssc = StandardScaler()
    X_train = ssc.fit_transform(X_train)
    X_test = ssc.transform(X_test)
    classifier = KerasClassifier(build_fn = build_classifier,  batch_size = 10, epochs = 10)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
    return accuracies
