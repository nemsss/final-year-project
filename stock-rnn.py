# Project

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
TRAINING_FILE_DIR = ''
training_set = pd.read_csv(TRAINING_FILE_DIR)
training_set = training_set.iloc[:,7:8].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[0:19749]
y_train = training_set[1:19750]

# Reshaping
X_train = np.reshape(X_train, (19749, 1, 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 28, epochs = 200)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of year X
TEST_FILE_DIR = ''
test_set = pd.read_csv(TEST_FILE_DIR)
stock_prices = test_set.iloc[0:4724,7:8].values

######################################################################################################################

x_begin = 0
x_end = 197
y_begin = 1
y_end = 198
stocks = [] ##Stock names

import math
from sklearn.metrics import mean_squared_error
    
for i in range(25):
   
    stock_prices = test_set.iloc[x_begin:x_end,7:8].values

    # Getting the predicted stock price of 2017
    inputs = stock_prices
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (197, 1, 1))
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    real_stock_price = test_set.iloc[x_begin:x_end,7:8].values
    
    # Visualising the results
    GRAPH_DIR = ''
    plt.plot(real_stock_price, color = 'red', label = 'Actual ' + stocks[i])
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ' + stocks[i])
    plt.title(stocks[i] +' Stock Price Prediction')
    plt.xlabel('Days from START_DATE to END_DATE')
    plt.ylabel('Prices')
    plt.legend()
    plt.savefig(GRAPH_DIR +stocks[i], orientation='landscape')
    plt.show()
    
    x_begin+= 198
    x_end+= 198
    y_begin+= 198
    y_end+= 198   

    # Evaluating the RNN   
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    print (rmse)
