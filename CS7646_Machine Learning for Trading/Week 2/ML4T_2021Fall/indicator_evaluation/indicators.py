import matplotlib
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt


def get_simple_moving_average(df_prices, lookback):
    normalized_prices = df_prices/df_prices.iloc[0]
    sma = normalized_prices.rolling(window=lookback).mean()
    return sma


def get_bb(df_prices, lookback):
    normalized_prices = df_prices / df_prices.iloc[0]
    sma = normalized_prices.rolling(window=lookback).mean()
    sd = normalized_prices.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * sd)
    lower_band = sma - (2 * sd)
    BBP = (normalized_prices - lower_band) / (top_band - lower_band)
    return BBP, top_band, lower_band



def get_momentum(df_prices, lookback):
    normalized_prices = df_prices / df_prices.iloc[0]
    df_momentum = (normalized_prices / normalized_prices.shift(periods=10)) - 1
    return df_momentum


def get_macd(df_prices, normalized_prices):
    ema1 = normalized_prices.ewm(span=12, adjust=False).mean()
    ema2 = normalized_prices.ewm(span=26, adjust=False).mean()
    macd = ema1 - ema2
    ema3 = macd.ewm(span=9, adjust=False).mean()
    figure4, axis = plt.subplots()
    plt.xticks(rotation=12)
    axis.set(xlabel='Date', ylabel="Price", title="MACD and MACD Signal")
    axis.plot(macd, label="MACD")
    #axis.plot(normalized_prices, label="JPM")
    axis.plot(ema3, label="MACD Signal Line")
    plt.axhline(y=0.0, color='r', linestyle='--')
    axis.grid(True)
    axis.legend()
    figure4 = matplotlib.pyplot.gcf()
    figure4.set_size_inches(14.5, 4, forward=True)
    figure4.savefig('4]MACD.png', dpi=100)
    plt.clf()
    figure4a, axis = plt.subplots()
    plt.xticks(rotation=12)
    axis.set(xlabel='Date', ylabel="Price", title="JPM Adj. Closing Price")
    axis.plot(df_prices, label="JPM Price")
    #axis.plot(df_momentum, label="Momentum")
    #plt.axhline(y=0.0, color='r', linestyle='--')
    axis.grid(True)
    axis.legend()
    figure4a = matplotlib.pyplot.gcf()
    figure4a.set_size_inches(14.5, 5, forward=True)
    figure4a.savefig('4a]JPMPrice.png', dpi=100)
    plt.clf()


def get_volatility(normalized_prices, df_prices, lookback):
    volatility = df_prices.rolling(window=lookback,center=False).std()
    volatility = volatility
    volatility = volatility.dropna()
    figure5, axis = plt.subplots()
    plt.xticks(rotation=12)
    axis.set(xlabel='Date', ylabel="Price(Normalized)", title="Volatility")
    axis.plot(normalized_prices, label="Normalized Price")
    axis.plot(volatility, label="Volatility")
    axis.grid(True)
    axis.legend()
    figure5 = matplotlib.pyplot.gcf()
    figure5.set_size_inches(14.5, 5, forward=True)
    figure5.savefig('5]Volatility.png',dpi=100)
    plt.clf()


def author():
    return "mlukacsko3"


def test_code():
    lookback = 14
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    syms = ['JPM']
    dates = pd.date_range(sd, ed)
    df_prices = get_data(syms, dates)
    df_prices = df_prices[syms]
    normalized_prices = df_prices / df_prices.iloc[0, :]
    sma = get_simple_moving_average(df_prices, lookback)
    get_bb(sma, normalized_prices, df_prices, lookback)
    get_momentum(df_prices, normalized_prices, lookback)
    get_volatility(normalized_prices, df_prices, lookback)
    get_macd(df_prices, normalized_prices)



if __name__ == "__main__":
    test_code()