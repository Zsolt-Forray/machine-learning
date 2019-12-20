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
__date__    = '19/12/2019'
__status__  = 'Development'


import sys
import os
import argparse

TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR_NAME = "machine_learning"
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, PROJECT_DIR_NAME))

sys.path.insert(0, PROJECT_DIR)

from machine_learning_model import MLModel
from feature_generator import Feature


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
            "model",
            type=str,
            default="knn",
            help="K-NN and SVM supervised machine learning classification algorithms"
    )
    parser.add_argument(
            "ticker",
            type=str,
            default="AMAT",
            help="ticker symbol"
    )
    parser.add_argument(
            "--features",
            nargs="+",
            type=int,
            default=[2,5,12],
            help="features combinations, max. 3 number between 1 and 17"
    )
    parser.add_argument(
            "--k",
            nargs="+",
            type=int,
            default=5,
            help="number of neighbors"
    )
    parser.add_argument(
            "--C",
            nargs="+",
            type=int,
            default=40,
            help="parameter C of RBF kernel SVM"
    )
    parser.add_argument(
            "--gamma",
            nargs="+",
            type=int,
            default=5,
            help="parameter gamma of RBF kernel SVM"
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
            k = args.k[0]
            avg_acc = ml_obj.run_knn(k)
        elif model == "svm":
            C = args.C[0]
            gamma = args.gamma[0]
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
