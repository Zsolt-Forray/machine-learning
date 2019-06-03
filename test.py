import unittest
import machine_learning_model as ml
from machine_learning_model import InvalidParamsError


class TestMLM(unittest.TestCase):
    def test_knn_3features(self):
        self.assertEqual(ml.run_knn(ticker="AMAT", k=5, features_combinations=(2,5,12)), 50.6)

    def test_knn_2features(self):
        self.assertEqual(ml.run_knn(ticker="MU", k=7, features_combinations=(1,8)), 47.6)

    def test_knn_1feature(self):
        self.assertEqual(ml.run_knn(ticker="MU", k=7, features_combinations=(8)), None)

    def test_knn_4features(self):
        self.assertEqual(ml.run_knn(ticker="MU", k=7, features_combinations=(8,9,12,1)), None)

    def test_knn_invalid_k(self):
        with self.assertRaises(InvalidParamsError):
            ml.run_knn(ticker="MU", k=21, features_combinations=(8,9,12))

    def test_invalid_ticker(self):
        self.assertEqual(ml.check_params(ticker="KO", features_combinations=(8,9,12)), False)

    def test_features_outside(self):
        self.assertEqual(ml.check_params(ticker="MU", features_combinations=(8,9,19)), False)

    def test_svm_3features(self):
        self.assertEqual(ml.run_svm(ticker="MSFT", C=40, gamma=5, features_combinations=(7,5,17)), 55.0)

    def test_svm_2features(self):
        self.assertEqual(ml.run_svm(ticker="JD", C=40, gamma=50, features_combinations=(16,10)), 58.4)

    def test_svm_1feature(self):
        self.assertEqual(ml.run_svm(ticker="JD", C=40, gamma=50, features_combinations=(10)), None)

    def test_svm_4features(self):
        self.assertEqual(ml.run_svm(ticker="JD", C=40, gamma=50, features_combinations=(1,2,3,4)), None)

    def test_svm_invalid_gamma(self):
        with self.assertRaises(InvalidParamsError):
            ml.run_svm(ticker="JD", C=40, gamma=500, features_combinations=(1,2,4))


if __name__ == "__main__":
    unittest.main()
