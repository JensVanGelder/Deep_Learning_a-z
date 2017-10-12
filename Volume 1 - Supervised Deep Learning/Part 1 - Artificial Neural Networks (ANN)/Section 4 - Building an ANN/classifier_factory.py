# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:16:20 2017

@author: jens
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
 
def build_classifier_gridsearch(optimizer, h1_size, h2_size):
    classifier = Sequential()
    classifier.add(Dense(input_dim = 11,
                         units = h1_size, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = h2_size, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier