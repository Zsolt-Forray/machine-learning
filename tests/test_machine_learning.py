#!/usr/bin/python3


"""
Test for machine learning analyzing tool
"""


__author__  = 'Zsolt Forray'
__license__ = 'MIT'
__version__ = '0.0.1'
__date__    = '20/12/2019'
__status__  = 'Development'


import sys
import os
import unittest

TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR_NAME = "machine_learning"
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", PROJECT_DIR_NAME))

sys.path.insert(0, PROJECT_DIR)

from machine_learning_model import MLModel
from feature_generator import Feature


class TestFeature(unittest.TestCase):
    def setUp(self):
        ticker = "AMAT"
        features_combinations = (2,5,12)
        base_path = os.path.dirname(os.path.abspath(__file__))
        quotes_path = os.path.join(base_path, "..", "DailyQuotes/{}.txt")
        self.fgen = Feature(ticker, features_combinations, quotes_path)

    def test_read_quotes(self):
        # Verify the quotes are loaded
        self.assertIsNotNone(self.fgen.read_quotes())

    def test_get_data_columns(self):
        stock_data_arr = self.fgen.read_quotes()
        self.fgen.get_data_columns(stock_data_arr)
        # Verify the first price and volume data
        self.assertAlmostEqual(32.36, self.fgen.open_price_arr[0], places=2)
        self.assertAlmostEqual(32.71, self.fgen.high_price_arr[0], places=2)
        self.assertAlmostEqual(31.66, self.fgen.low_price_arr[0], places=2)
        self.assertAlmostEqual(31.94, self.fgen.close_price_arr[0], places=2)
        self.assertEqual(13209400, self.fgen.volume_arr[0])

    def test_generate_labels(self):
        stock_data_arr = self.fgen.read_quotes()
        self.fgen.get_data_columns(stock_data_arr)
        self.assertIsNotNone(self.fgen.generate_labels())

    def test_generate_features(self):
        stock_data_arr = self.fgen.read_quotes()
        self.fgen.get_data_columns(stock_data_arr)
        self.assertIsNotNone(self.fgen.generate_features())


if __name__ == "__main__":
    unittest.main()
