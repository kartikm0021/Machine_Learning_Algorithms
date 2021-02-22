import numpy as np
import csv
import time
from sklearn.metrics import roc_curve
from matplotlib import pyplot


class Linear_Regression:
    def __init__(self, X, y, alpha=0.03, iterations=1500):

        self.alpha = alpha
        self.iterations = iterations
        self.total_number_of_records = len(y)
        self.n_features = np.size(X, 1)
        self.normalize_and_add_coefficeints(X, y)
        # self.X = np.hstack((np.ones(
        #     (self.total_number_of_records, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        # self.y = y[:, np.newaxis]
        # self.weights = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

    def dataLoad(self, fileName):
        """
        dataLoad loads data from filename provided into a numpy array X and returns it.

        :filename filename: File name to be loaded to a numpy nd array object.
        :return: numpy array object containing the entire data set
        """
        print(f'Loading file {fileName}')
        data = np.genfromtxt(
            fileName)
        print(f'Loaded file {fileName} with {data.shape} records')
        return data

    def normalize_and_add_coefficeints(self, X, y):
        self.X = np.hstack(
            (self.add_intercept_coefficient(), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.weights = np.zeros((self.n_features + 1, 1))

    def add_intercept_coefficient(self):
        return np.ones((self.total_number_of_records, 1))

    def update_weights(self):
        self.weights = self.weights - \
            (self.alpha/self.total_number_of_records) * \
            self.X.T @ (self.y_hat() - self.y)

    def y_hat(self):
        return self.X @ self.weights

    def model(self):

        for _ in range(self.iterations):
            self.update_weights()

        self.intercept_ = self.weights[0]
        self.coef_ = self.weights[1:]

        return self

    def accuracy(self, X=None, y=None):

        if X is None:
            X = self.X
        else:
            total_number_of_records = np.size(X, 0)
            X = np.hstack((np.ones(
                (total_number_of_records, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.weights
        score = self.calculate_coeeficient_r2_score(y, y_pred)

        return score

    def calculate_coeeficient_r2_score(self, y_true, y_pred):
        score = 1 - (((y_true - y_pred)**2).sum() /
                     np.sum(((y_true - np.mean(y_true))**2)))
        return score

    def predict(self, X):
        total_number_of_records = np.size(X, 0)
        y = np.hstack((np.ones((total_number_of_records, 1)), (X-np.mean(X, 0))
                       / np.std(X, 0))) @ self.weights
        return y

    def get_weights(self):

        return self.weights

    @classmethod
    def splitTT(cls, X, percentTrain):
        """
        Takes in the normalized dataset X_norm , and the expected portion
        of train dataset percentTrain (e.g. 0.6), returns a list X_split=[X_train,X_test]
        :X: nd array object normalized.
        :percentTrain: percent of the records which are to be splitted to train and test.
        :return: list of numpy array objects containing the Training and Test records
        """
        np.random.shuffle(X)
        N = len(X)
        sample = int(percentTrain*N)
        x_train, x_test = X[:sample, :], X[sample:, :]
        return [x_train, x_test]


# https://towardsdatascience.com/linear-regression-from-scratch-with-numpy-implementation-finally-8e617d8e274c
