"""
Author: Tim Campbell
Modified: 3/24/2021
Notes: Implements a sequential neural network using keras.
"""


import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from config import cols
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=3, suppress=True)


def main():
    csv_files = [f for f in listdir('StockCSVRecs') if isfile(join("StockCSVRecs", f))]
    os.chdir('StockCSVRecs')
    for csv_name in csv_files:
        df = pd.read_csv(csv_name, usecols=cols)
        # date = pandas.read_csv(csv_name, usecols=[0])
        stock_name = "$" + csv_name[:-8]
        #train_data = df.copy()

        # Single column output
        #output_data = pd.DataFrame({"Close": df.pop("Close")})
        neural_network(stock_name, df)


        # Double column output
        # output_data = pandas.DataFrame()
        # output_data["Low"] = data.pop("Low")
        # output_data["High"] = data.pop("High")
    os.chdir('..')


def neural_network(stock_name, df):

    #normalize = tf.keras.layers.experimental.preprocessing.Normalization()
    
    X = df.drop('Close', axis=1)
    y = np.ravel(df['Close'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    

    #train_data = np.array(train_data)

    #normalize.adapt(train_data)

    stock_model = tf.keras.Sequential([
        #normalize,
        layers.Dense(32),
        layers.Dense(64),
        layers.Dense(1)])

    stock_model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=tf.keras.metrics.Accuracy())

    # calculate num of epochs based on length of dataset
    weighted_epoch = round(len(df)/100)
    print("-----Beginning training for %s-----" % stock_name)

    stock_model.fit(X_train, y_train, epochs=weighted_epoch)

    prediction = stock_model.predict(X_test, batch_size=None, verbose=0, steps=1, callbacks=None,
                                     max_queue_size=10, workers=1, use_multiprocessing=False)
    
    prediction_price = float(str(prediction[-1]).strip("[").strip("]"))


    
    print()
    print("Same day price prediction for %s: " % stock_name, (str(round(prediction_price, 2))))
    print("Actual price for %s: $" % stock_name + str(round(y_test[-1], 2)))
    print()

    plt.plot(prediction, label='Predicted')
    plt.plot(y_test, label='Actual')
    # plt.plot(output_data)
    plt.title(stock_name)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()
