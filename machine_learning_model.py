"""
----------------------------------------------------------------------------------
                            MACHINE LEARNING MODEL
----------------------------------------------------------------------------------

The K-NN and SVM supervised machine learning classification algorithms can be used
to predict whether the stock opens higher than the previous day closing price.

This script returns the Prediction Accuracy of K-NN or SVM models.

|
| Input parameter(s):
|                   KNN:    Ticker Symbol, K, Features Combinations (max.3)
|                           eg. "AMAT", 5, (2, 5, 12)
|
|                   SVM:    Ticker Symbol, C, Gamma, Features Combinations (max.3)
|                           eg. "AMAT", 40, 5, (2, 5, 12)

Available Ticker Symbols: (AMAT, C, JD, MSFT, MU, TWTR)

Valid Machine Learning Algorithms: (KNN, SVM)

Valid Features applied for predictive model:    Group of integers from 1 to 17
                                                Maximum 3 features can be selected

Features are generated from the OHLCV data of the last 2 trading days.
They are derived as a ratio of different OHLCV data eg.: Close (1) / Close (2).
Index (1) - the day prior the predicted day
Index (2) - 2 days prior the predicted day

These features are used to predict whether the open price of the next day
will be above or below the last close price.

       1: Close (1) : Open (1)
       2: Close (2) : Open (2)
       3: High (1) : Low (1)
       4: High (2) : Low (2)
       5: Volume (1) : Volume (2)
       6: Body (1) : Body (2)
       7: Range (1) : Range (2)
       8: Low (1) : Low (2)
       9: High (1) : High (2)
       10: Close (1) : Close (2)
       11: Open (1) : Open (2)
       12: Close (1) : Open (2)
       13: Open (1) : Close (2)
       14: High (1) : Middle of Body (1)
       15: High (2) : Middle of Body (2)
       16: High (1) : Middle of Range (1)
       17: High (2) : Middle of Range (2)

Remark: Input parameters must be separated by comma(s).

----------------------------------------------------------------------------------
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm
from sklearn import metrics
import numpy as np
import re

import feature_generator


class InvalidTickersError(Exception):
    pass

class InvalidFeaturesError(Exception):
    pass

class InvalidParamsError(Exception):
    pass

def train_test_data(train_index, test_index, X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

def preprocessing_train(X_train):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_X_train = scaler.fit_transform(X_train)
    return scaled_X_train

def preprocessing_test(X_test, X_train):
    scaled_X_test = (X_test - X_train.min(axis=0))\
                    / (X_train.max(axis=0) - X_train.min(axis=0))
    return scaled_X_test

# Cross-Validation (K-fold)
def k_fold(model, X, y):
    acc_list = []
    kf = KFold(n_splits=10, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = train_test_data(train_index, test_index, X, y)
        scaled_X_train = preprocessing_train(X_train)
        scaled_X_test = preprocessing_test(X_test, X_train)
        model.fit(scaled_X_train, y_train)
        y_predict = model.predict(scaled_X_test)

        # Accuracy per folds
        # These two calculations give the same result
        # accuracy = model.score(scaled_X_test, y_test)
        acc_met = metrics.accuracy_score(y_test, y_predict)
        acc_list.append(acc_met)

    # Average accuracy of all the folds
    avg_acc = np.mean(acc_list)
    return avg_acc

def calc(ticker, selected_model, features_combinations):
    X, y = feature_generator.run_fgen(ticker, features_combinations)
    avg_acc = k_fold(selected_model, X, y)
    return avg_acc

def check_params(ticker, features_combinations):
    val_tickers = ("AMAT", "C", "JD", "MSFT", "MU", "TWTR")
    try:
        ticker = str.upper(ticker)

        if ticker not in val_tickers:
            raise InvalidTickersError()

        elif min(features_combinations) < 1 or max(features_combinations) > 17 \
            or len(features_combinations) > 3:
            raise InvalidFeaturesError()
        return True

    except (InvalidTickersError, InvalidFeaturesError,
            InvalidParamsError, KeyError, ValueError, TypeError):
        print("[Error] Invalid input paramater(s)")
        return False

def run_knn(ticker, k, features_combinations):
    res_bool = check_params(ticker, features_combinations)
    if res_bool == True:
        if k not in range(3,16,2):
            raise InvalidParamsError()
        selected_model = knn(n_neighbors= k)

        avg_acc = calc(ticker, selected_model, features_combinations)
        return round(avg_acc * 100,2)

def run_svm(ticker, C, gamma, features_combinations):
    res_bool = check_params(ticker, features_combinations)
    if res_bool == True:
        if C not in range(1,101) or gamma not in range(1,101):
            raise InvalidParamsError()
        selected_model = svm.SVC(kernel= "rbf", C= C, gamma= gamma)

        avg_acc = calc(ticker, selected_model, features_combinations)
        return round(avg_acc * 100,2)
