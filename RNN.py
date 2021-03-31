<<<<<<< HEAD
"""import numpy as np
=======
import os
from os import listdir

import numpy as np
>>>>>>> fd3dafe1b47702de590f2fabb00c17236551cdaf
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from os.path import isfile, join


def main():
<<<<<<< HEAD
    # Graph modeling test:
    # sin_wave = np.array([math.sin(x) for x in np.arange(200)])
    # plt.plot(sin_wave[:100])
    # plt.show()

    df = pandas.read_csv("Google.csv")
    dataset = df.values

    X = dataset[:, 1:6]
    Y = dataset[:, 1]

    min_max_scaler = preprocessing.MinMaxScaler()
    XScale = min_max_scaler.fit_transform(X)

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(XScale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    X_train = np.asarray(X_train).astype(np.float32)
    Y_train = np.asarray(Y_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    Y_val = np.asarray(Y_val).astype(np.float32)

    # print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
#
#    model = Sequential([Dense(32, activation='relu', input_shape=(5, )), Dense(32, activation='relu'),
#                        Dense(1, activation='sigmoid'), ])
#    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#    hist = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_val, Y_val))

def rnn():
    dataset = pandas.read_csv("Google.csv", usecols=[0, 1, 2, 3, 4, 5])
    dataset.head()
    train = dataset.loc[:, ['Open']].values
    scaled_data = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaled_data.fit_transform(train)
    plt.plot(train_scaled)
    plt.show() 

if __name__ == '__main__':
   rnn() """ 
import numpy as np 
import tensorflow as tf 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow import keras 
from tensorflow.keras import layers 
import math 
import yfinance as yf 
import matplotlib.pyplot as plt 
import pandas 
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from keras.layers import SimpleRNN 
from keras.layers import Dropout 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 

#imports config epoch line
from config import epoch

=======
    csv_files = [f for f in listdir('StockCSV') if isfile(join("StockCSV", f))]
    for csv_name in csv_files:
        os.chdir('StockCSV')
        f_in = open(csv_name, 'r')
        csv_header = f_in.readline()
        csv_header_list = list(csv_header.split(','))
        csv_header_list = csv_header_list[1:5]
        for value in csv_header_list:
            print(csv_name, value)
            rnn(str(csv_name), str(value))
        os.chdir('..')


def rnn(csv_name, value):
    dataset = pandas.read_csv(csv_name, usecols=[0, 1, 2, 3, 4, 5])
    dataset.head()
    train = dataset.loc[:, [value]].values

    scaled_data = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaled_data.fit_transform(train)
    plt.plot(train_scaled)

    X_train = []
    Y_train = []

    time_steps = 50

    for i in range(time_steps, 1250):
        X_train.append(train_scaled[i - time_steps:i, 0])
        Y_train.append(train_scaled[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Initialize RNN:
    regressor = Sequential()

    # Adding the first RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding the second RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding the third RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding the fourth RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units=1))

    # Compile the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, Y_train, epochs=3, batch_size=32)
>>>>>>> fd3dafe1b47702de590f2fabb00c17236551cdaf


def rnn():
    dataset = pandas.read_csv("Google.csv", usecols=[0, 1, 2, 3, 4, 5])
    dataset.head()
    train = dataset.loc[:, ['Open']].values
    scaled_data = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaled_data.fit_transform(train)
    plt.plot(train_scaled)
    X_train = []
    Y_train = []
    time_steps = 50
    for i in range(time_steps, 1250):
        X_train.append(train_scaled[i - time_steps:i, 0])
        Y_train.append(train_scaled[i, 0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Initialize RNN:
    regressor = Sequential()
    # Adding the first RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Adding the second RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
    regressor.add(Dropout(0.2))
    # Adding the third RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
    regressor.add(Dropout(0.2))
    # Adding the fourth RNN layer and some Dropout regularization
    regressor.add(SimpleRNN(units=50))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units=1))
    # Compile the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    # Fitting the RNN to the Training set
    regressor.fit(X_train, Y_train, epochs=epoch, batch_size=32) 
if __name__ == '__main__':
    main()
