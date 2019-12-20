#!/usr/bin/python3

"""
----------------------------------------------------------------------------------
                            MACHINE LEARNING MODEL
----------------------------------------------------------------------------------

The K-NN and SVM supervised machine learning classification algorithms can be used
to predict whether the stock opens higher than the previous day closing price.

This script returns the Prediction Accuracy of K-NN or SVM models.

| Input parameter(s):
|                   KNN:    Ticker, Features Combinations (max.3) -> K
|                           eg. "AMAT", (2, 5, 12) -> 5
|
|                   SVM:    Ticker, Features Combinations (max.3) -> C, Gamma
|                           eg. "AMAT", (2, 5, 12) -> 40, 5

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


__author__  = 'Zsolt Forray'
__license__ = 'MIT'
__version__ = '0.0.1'
__date__    = '20/12/2019'
__status__  = 'Development'


import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm
from sklearn import metrics
from feature_generator import Feature
from user_defined_exceptions import InvalidTickersError
from user_defined_exceptions import InvalidFeaturesError
from user_defined_exceptions import InvalidParamsError


class MLModel:
    def __init__(self, ticker, features_combinations):
        self.ticker = ticker.upper()
        self.combinations = features_combinations
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.quotes_path = os.path.join(base_path, "..", "DailyQuotes/{}.txt")

    @staticmethod
    def train_test_data(train_index, test_index, X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def preprocessing_train(X_train):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_X_train = scaler.fit_transform(X_train)
        return scaled_X_train

    @staticmethod
    def preprocessing_test(X_test, X_train):
        scaled_X_test = (X_test - X_train.min(axis=0))\
                        / (X_train.max(axis=0) - X_train.min(axis=0))
        return scaled_X_test

    @staticmethod
    def k_fold(model, X, y):
        # Cross-Validation (K-fold)
        acc_list = []
        kf = KFold(n_splits=10, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = \
            MLModel.train_test_data(train_index, test_index, X, y)
            scaled_X_train = MLModel.preprocessing_train(X_train)
            scaled_X_test = MLModel.preprocessing_test(X_test, X_train)
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

    def calc(self, model):
        fgen_obj = Feature(self.ticker, self.combinations, self.quotes_path)
        X, y = fgen_obj.run_fgen()
        avg_acc = MLModel.k_fold(model, X, y)
        return avg_acc

    def check_ticker(self):
        if not os.path.isfile(self.quotes_path.format(self.ticker)):
            raise InvalidTickersError()

    def check_features(self):
        if isinstance(self.combinations, int):
            raise InvalidFeaturesError()
        elif len(self.combinations) > 3 or min(self.combinations) < 1 \
             or max(self.combinations) > 17:
            raise InvalidFeaturesError()

    @staticmethod
    def check_model(model_name, *args):
        if model_name == "run_knn" and args[0] not in range(3,16,2):
            # k parameter = args[0]
            raise InvalidParamsError()
        elif model_name == "run_svm" and \
                        (args[0] not in range(1,101) or args[1] not in range(1,101)):
            # C, gamma parameters = args[0], args[1]
            raise InvalidParamsError()

    def check_params(self, model_name, *args):
        try:
            self.check_ticker()
            self.check_features()
            MLModel.check_model(model_name, *args)
            return True
        except (InvalidTickersError, InvalidFeaturesError, InvalidParamsError):
            print("[Error] Invalid input paramater(s)")
            return False

    def run_knn(self, k):
        """
        Returns the prediction accuracy of KNN model
        """
        model_name = self.run_knn.__name__
        valid_params = self.check_params(model_name, k)
        if valid_params:
            model = knn(n_neighbors=k)
            avg_acc = self.calc(model)
            return round(avg_acc * 100,2)

    def run_svm(self, C, gamma):
        """
        Returns the prediction accuracy of SVM model
        """
        model_name = self.run_svm.__name__
        valid_params = self.check_params(model_name, C, gamma)
        if valid_params:
            model = svm.SVC(kernel="rbf", C=C, gamma=gamma)
            avg_acc = self.calc(model)
            return round(avg_acc * 100,2)


if __name__ == "__main__":
    ml_obj = MLModel(ticker="AMAT", features_combinations=(2,5,12))
    ml_obj.run_knn(k=5)
    ml_obj.run_svm(C=40, gamma=5)
