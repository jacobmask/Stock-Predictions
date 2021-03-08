import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas


def main():
    sin_wave = np.array([math.sin(x) for x in np.arange(200)])

    plt.plot(sin_wave[:100])
    plt.show()

    df = pandas.read_csv("Google.csv")
    print(df)


if __name__ == '__main__':
    main()
