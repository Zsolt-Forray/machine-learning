#!/usr/bin/python3


"""
Features applied for predictive model construction are derived from stock OHLCV data.
"""


__author__  = 'Zsolt Forray'
__license__ = 'MIT'
__version__ = '0.0.1'
__date__    = '19/12/2019'
__status__  = 'Development'


import numpy as np
import os


class Feature:
    def __init__(self, ticker, features_combinations):
        self.ticker = ticker
        self.combinations = features_combinations

    @staticmethod
    def define_path():
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, "..", "DailyQuotes/{}.txt")

    def read_quotes(self):
        db_path = Feature.define_path()
        raw_stock_quotes = np.loadtxt(db_path.format(self.ticker),
                                      skiprows=1, usecols=(1,2,3,4,6), delimiter=",")
        stock_data_arr = raw_stock_quotes.T
        return stock_data_arr

    def get_data_columns(self, stock_data_arr):
        self.open_price_arr = stock_data_arr[0]
        self.high_price_arr = stock_data_arr[1]
        self.low_price_arr = stock_data_arr[2]
        self.close_price_arr = stock_data_arr[3]
        self.volume_arr = stock_data_arr[4]

    def generate_labels(self):
        # label is (+1) if next day the stock opens higher
        # label is (-1) if next day the stock opens lower or at the last close price
        open_prev_close_arr = (np.roll(self.open_price_arr,-2) \
                               - np.roll(self.close_price_arr,-1))[:-2]
        label_arr = np.where(open_prev_close_arr > 0, 1, -1)
        return label_arr

    def gen_features_ohlcv_1(self):
        # Index #1: the day prior the predicted day
        # OHLCV data #1
        open1   = np.roll(self.open_price_arr, -1)[:-2]
        high1   = np.roll(self.high_price_arr, -1)[:-2]
        low1    = np.roll(self.low_price_arr, -1)[:-2]
        close1  = np.roll(self.close_price_arr, -1)[:-2]
        volume1 = np.roll(self.volume_arr, -1)[:-2]
        return open1, high1, low1, close1, volume1

    @staticmethod
    def gen_features_derived_ohlcv_1(open1, high1, low1, close1, volume1):
        # Derived from OHLC data #1
        body1      = close1 - open1
        body1      = np.where(body1 == 0, 0.01, body1)
        range1     = high1 - low1
        range1     = np.where(range1 == 0, 0.01, range1)
        mid_body1  = (open1 + close1) / 2
        mid_range1 = (low1 + high1) / 2
        return body1, range1, mid_body1, mid_range1

    def gen_features_ohlcv_2(self):
        # Index #2: 2 days prior the predicted day
        # OHLCV data #2
        open2   = self.open_price_arr[:-2]
        high2   = self.high_price_arr[:-2]
        low2    = self.low_price_arr[:-2]
        close2  = self.close_price_arr[:-2]
        volume2 = self.volume_arr[:-2]
        return open2, high2, low2, close2, volume2

    @staticmethod
    def gen_features_derived_ohlcv_2(open2, high2, low2, close2, volume2):
        # Derived from OHLC data #2
        body2      = close2 - open2
        body2      = np.where(body2 == 0, 0.01, body2)
        range2     = high2 - low2
        range2     = np.where(range2 == 0, 0.01, range2)
        mid_body2  = (open2 + close2) / 2
        mid_range2 = (low2 + high2) / 2
        return body2, range2, mid_body2, mid_range2

    def generate_features(self):
        """
        Features are generated from the OHLCV data (components) of the last 2 trading days.
        Features are derived as a ratio of different OHLCV data eg.: close1 / close2.
        These features are then used to predict whether the open price of the next day
        will be above or below the last close price.
        """

        open1, high1, low1, close1, volume1 = self.gen_features_ohlcv_1()
        body1, range1, mid_body1, mid_range1 = \
        Feature.gen_features_derived_ohlcv_1(open1, high1, low1, close1, volume1)
        open2, high2, low2, close2, volume2 = self.gen_features_ohlcv_2()
        body2, range2, mid_body2, mid_range2 = \
        Feature.gen_features_derived_ohlcv_2(open2, high2, low2, close2, volume2)

        features_dict ={
                        1:(close1, open1),
                        2:(close2, open2),
                        3:(high1, low1),
                        4:(high2, low2),
                        5:(volume1, volume2),
                        6:(body1, body2),
                        7:(range1, range2),
                        8:(low1, low2),
                        9:(high1, high2),
                        10:(close1, close2),
                        11:(open1, open2),
                        12:(close1, open2),
                        13:(open1, close2),
                        14:(high1, mid_body1),
                        15:(high2, mid_body2),
                        16:(high1, mid_range1),
                        17:(high2, mid_range2),
                    }
        # Features: ratio is calculated from data in 'features_dict'
        selected_features_arr = np.zeros((len(open1),len(self.combinations)))
        for i, comb in enumerate(self.combinations):
            # FEATURES as ratio
            selected_features_arr[:,i] = features_dict[comb][0] / features_dict[comb][1]
        return selected_features_arr

    def run_fgen(self):
        stock_data_arr = self.read_quotes()
        self.get_data_columns(stock_data_arr)
        label_arr = self.generate_labels()
        selected_features_arr = self.generate_features()
        return selected_features_arr, label_arr
