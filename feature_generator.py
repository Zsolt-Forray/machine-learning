"""
Features applied for predictive model construction are derived from stock OHLCV data.
"""
import numpy as np
import os

# Path settings
base_path = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_path, "DailyQuotes/{}.txt")

def read_quotes(ticker):
    raw_stock_quotes = np.loadtxt(db_path.format(ticker),
                        skiprows=1, usecols=(1,2,3,4,6), delimiter=",")
    stock_data_arr = raw_stock_quotes.T
    return stock_data_arr

def label_generator(open_price_arr, close_price_arr):
    # label is (+1) if next day the stock opens higher
    # label is (-1) if next day the stock opens lower or at the last close price
    open_prev_close_arr = (np.roll(open_price_arr,-2) - np.roll(close_price_arr,-1))[:-2]
    label_arr = np.where(open_prev_close_arr > 0, 1, -1)
    return label_arr

def feature_generator(stock_data_arr, combinations):
    # Features are generated from the OHLCV data of the last 2 trading days.
    # They are derived as a ratio of different OHLCV data eg.: close1 / close2.
    # These features are used to predict whether the open price of the next day
    # will be above or below the last close price.

    open_price_arr = stock_data_arr[0]
    high_price_arr = stock_data_arr[1]
    low_price_arr = stock_data_arr[2]
    close_price_arr = stock_data_arr[3]
    volume_arr = stock_data_arr[4]

    # Index #1: the day prior the predicted day
    # OHLCV data
    open1 = np.roll(open_price_arr, -1)[:-2]
    high1 = np.roll(high_price_arr, -1)[:-2]
    low1 = np.roll(low_price_arr, -1)[:-2]
    close1 = np.roll(close_price_arr, -1)[:-2]
    volume1 = np.roll(volume_arr, -1)[:-2]

    # Derived from OHLC
    body1 = close1 - open1
    body1 = np.where(body1 == 0, 0.01, body1)
    range1 = high1 - low1
    range1 = np.where(range1 == 0, 0.01, range1)
    mid_body1 = (open1 + close1) / 2
    mid_range1 = (low1 + high1) / 2

    # Index #2: 2 days prior the predicted day
    # OHLCV data
    open2 = open_price_arr[:-2]
    high2 = high_price_arr[:-2]
    low2 = low_price_arr[:-2]
    close2 = close_price_arr[:-2]
    volume2 = volume_arr[:-2]

    # Derived from OHLC
    body2 = close2 - open2
    body2 = np.where(body2 == 0, 0.01, body2)
    range2 = high2 - low2
    range2 = np.where(range2 == 0, 0.01, range2)
    mid_body2 = (open2 + close2) / 2
    mid_range2 = (low2 + high2) / 2

    # Features: ratio is calculated from data in 'features_dict'
    features_dict ={
                    1:("close1_open1", close1, open1),
                    2:("close2_open2", close2, open2),
                    3:("high1_low1", high1, low1),
                    4:("high2_low2", high2, low2),
                    5:("vol12", volume1, volume2),
                    6:("body12", body1, body2),
                    7:("range12", range1, range2),
                    8:("low12", low1, low2),
                    9:("high12", high1, high2),
                    10:("close12", close1, close2),
                    11:("open12", open1, open2),
                    12:("close1_open2", close1, open2),
                    13:("open1_close2", open1, close2),
                    14:("high1_mid_body1", high1, mid_body1),
                    15:("high2_mid_body2", high2, mid_body2),
                    16:("high1_mid_range1", high1, mid_range1),
                    17:("high2_mid_range2", high2, mid_range2),
                }

    selected_features_arr = np.zeros((len(open1),len(combinations)))
    for i, c in enumerate(combinations):
        # FEATURES as ratio
        selected_features_arr[:,i] = features_dict[c][1] / features_dict[c][2]
    return selected_features_arr

def run_fgen(ticker, combinations):
    stock_data_arr = read_quotes(ticker)
    label_arr = label_generator(stock_data_arr[0], stock_data_arr[3])
    selected_features_arr = feature_generator(stock_data_arr, combinations)
    return selected_features_arr, label_arr
