# Machine Learning

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d62b4b06b5c7482f8140633379d42222)](https://www.codacy.com/app/forray.zsolt/machine-learning?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Zsolt-Forray/machine-learning&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/d62b4b06b5c7482f8140633379d42222)](https://www.codacy.com/manual/forray.zsolt/machine-learning?utm_source=github.com&utm_medium=referral&utm_content=Zsolt-Forray/machine-learning&utm_campaign=Badge_Coverage)
[![Build Status](https://travis-ci.com/Zsolt-Forray/machine-learning.svg?branch=master)](https://travis-ci.com/Zsolt-Forray/machine-learning)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Description
The K-NN and SVM supervised machine learning classification algorithms can be used to predict whether the stock opens higher than the previous day closing price.

Features applied for predictive model construction are derived from stock OHLCV data of the last 2 trading days.

The prediction accuracy is estimated by averaging the accuracies calculated in all the k cases of cross validation.

## Usage
This project uses sample stock quotes from the `DailyQuotes` folder. If you want, you can add other stock quotes to this folder. If you add stock quotes having different timeframe, do not forget to update the other quotes accordingly.

### Usage Example

**Parameters (K-NN):**

* ticker: Ticker symbol
* features_combinations: Group of integers from 1 to 17. Maximum 3 features can be selected.
* k: Number of neighbors

**Parameters (SVM):**

* ticker: Ticker symbol
* features_combinations: Group of integers from 1 to 17. Maximum 3 features can be selected.
* C: Parameter 'C' of 'RBF' kernel SVM
* gamma: Parameter 'gamma' of 'RBF' kernel SVM

**Features**

Derived as a ratio of different OHLCV data eg.: Close (1) / Close (2).  

* Index (1) - the day prior the predicted day
* Index (2) - 2 days prior the predicted day

These features are used to predict whether the open price of the next day will be above or below the last close price:  

1. Close (1) : Open (1)
2. Close (2) : Open (2)
3. High (1) : Low (1)
4. High (2) : Low (2)
5. Volume (1) : Volume (2)
6. Body (1) : Body (2)
7. Range (1) : Range (2)
8. Low (1) : Low (2)
9. High (1) : High (2)
10. Close (1) : Close (2)
11. Open (1) : Open (2)
12. Close (1) : Open (2)
13. Open (1) : Close (2)
14. High (1) : Middle of Body (1)
15. High (2) : Middle of Body (2)
16. High (1) : Middle of Range (1)
17. High (2) : Middle of Range (2)

#### Import package and call functions

```python
#!/usr/bin/python3

from machine_learning_model import MLModel

ml_obj = MLModel(ticker="AMAT", features_combinations=(2,5,12))

prediction_accuracy_knn = ml_obj.run_knn(k=5)
prediction_accuracy_svm = ml_obj.run_svm(C=40, gamma=5)

print(prediction_accuracy_knn, prediction_accuracy_svm)
```

**Output**

Returns the Prediction Accuracy (%) of K-NN or SVM models:  

* Prediction Accuracy of K-NN model: `prediction_accuracy_knn = 50.6 %`
* Prediction Accuracy of SVM model: `prediction_accuracy_svm = 58.6 %`

#### Using command-line interface

![Screenshot](/png/CLI.png)

## LICENSE
MIT

## Contributions
Contributions to this repository are always welcome.
This repo is maintained by Zsolt Forray (forray.zsolt@gmail.com).
