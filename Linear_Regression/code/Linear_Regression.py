import numpy as np
import csv
import time
from sklearn.metrics import roc_curve
from matplotlib import pyplot


class Linear_Regression:
    def __init__(self, X, y, alpha=0.03, n_iter=1500):

        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones(
            (self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
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

    def fit(self):

        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
                self.X.T @ (self.X @ self.params - self.y)

        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self

    def score(self, X=None, y=None):

        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())

        return score

    def predict(self, X):
        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0))
                       / np.std(X, 0))) @ self.params
        return y

    def get_params(self):

        return self.params

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

    def accuracy(self, y_true, y_prediction):
        """
        Function for calculating the accuracy given y true values and y prediction values
        :y_true: y true values.
        :y_prediction: y prediction values
        :return:accuracy values
        """
        accuracy = np.sum(y_true == y_prediction) / len(y_true)
        return accuracy
