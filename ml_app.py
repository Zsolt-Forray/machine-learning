#!/usr/bin/python3


"""
The K-NN and SVM supervised machine learning classification algorithms can be used \
to predict whether the stock opens higher than the previous day closing price. \
Features applied for predictive model construction are derived from \
stock OHLCV data of the last 2 trading days. The prediction accuracy is estimated \
by averaging the accuracies calculated in all the k cases of cross validation.
"""


__author__  = 'Zsolt Forray'
__license__ = 'MIT'
__version__ = '0.0.1'
__date__    = '20/12/2019'
__status__  = 'Development'


import sys
import os
import argparse

TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR_NAME = "machine_learning"
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, PROJECT_DIR_NAME))

sys.path.insert(0, PROJECT_DIR)

from machine_learning_model import MLModel


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, \
                      usage="python3 %(prog)s <model> <ticker> <features> [options]")
    parser.add_argument(
            "model",
            type=str,
            help="K-NN and SVM supervised machine learning classification algorithms, e.g., knn or svm"
    )
    parser.add_argument(
            "ticker",
            type=str,
            help="ticker symbol, e.g., AMAT"
    )
    parser.add_argument(
            "features",
            nargs="+",
            type=int,
            help="features combinations, max. 3 integers between 1 and 17, e.g., 2 5 12"
    )
    parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="KNN model: number of neighbors, one odd integer between 3 and 15, (default=5)"
    )
    parser.add_argument(
            "--C",
            type=int,
            default=40,
            help="SVM model: parameter C of RBF kernel SVM, one integer between 1 and 100, (default=40)"
    )
    parser.add_argument(
            "--gamma",
            type=int,
            default=5,
            help="SVM model: parameter gamma of RBF kernel SVM, one integer between 1 and 100, (default=5)"
    )
    args = parser.parse_args()
    return args

def main(args):
    try:
        model = args.model
        ticker = args.ticker
        features_combinations = [int(i) for i in args.features]
        ml_obj = MLModel(ticker, features_combinations)
        if model == "knn":
            k = args.k
            avg_acc = ml_obj.run_knn(k)
        elif model == "svm":
            C = args.C
            gamma = args.gamma
            avg_acc = ml_obj.run_svm(C, gamma)
        else:
            print("Info: Invalid model...")
            raise Exception()
    except Exception:
        print("Error: Check the input parameters...")
    else:
        if avg_acc:
            print("{} prediction accuracy: {}%".format(model.upper(), avg_acc))


if __name__ == "__main__":
    args = get_args()
    main(args)
