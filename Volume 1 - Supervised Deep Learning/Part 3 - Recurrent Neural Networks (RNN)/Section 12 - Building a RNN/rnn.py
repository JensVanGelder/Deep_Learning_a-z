# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:21:26 2017

@author: jens
"""
# Step 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing training set
training_set = pd.read_csv('Bolt_of_Damask_Train.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting inputs and outputs
X_train = training_set[0:1598]
y_train = training_set[1:1599]

# Reshaping
X_train = np.reshape(X_train, (1598, 1, 1))

# Step 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer ='adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Step 3 - Making predictions and visualisations

# Getting the real prices
test_set = pd.read_csv('Bolt_of_Damask_Test.csv')
real_price = test_set.iloc[:,1:2].values

# Predicting prices
inputs = real_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (13, 1, 1))

predicted_price = regressor.predict(inputs)
predicted_price = sc.inverse_transform(predicted_price)

# Visualising the results
plt.plot(real_price, color = 'red', label = 'Real Bolt of Damask Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Bolt of Damask Price')
plt.title('Bolts of Damask Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bolt of Damask Price')
plt.legend()
plt.show()

# Step 4 - Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_price, predicted_price))

