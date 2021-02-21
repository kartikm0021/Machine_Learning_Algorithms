import numpy as np
import csv
import time
from sklearn.metrics import roc_curve
from matplotlib import pyplot


class Linear_Regression:

    def __init__(self, fileName):
        self.fileName = fileName
        self.tolerance = 1e-5
        self.train_percentages = [0.6]
        self.cross_fold_values = [5]
        self.learning_rates = [.1, .2, .3, .4, 1]
        self.statistics = []

    def dataLoad(self):
        """
        dataLoad loads data from filename provided into a numpy array X and returns it.

        :filename filename: File name to be loaded to a numpy nd array object.
        :return: numpy array object containing the entire data set
        """
        print(f'Loading file {self.fileName}')
        data = np.genfromtxt(
            self.fileName)
        print(f'Loaded file {self.fileName} with {data.shape} records')
        return data

    def dataStandardization(self, data):
        number_of_columns = data.shape[1]
        for i in range(0, number_of_columns-1):
            v = data[:, i]
            mu = np.mean(v)
            sigma = np.std(v)
            normalized_column = (v - mu) / (sigma)
            data[:, i] = normalized_column
        return data

    def dataNorm(self, data):
        """
        DataNorm normalizes all the columns in the data set except the last column as the output is not to be normalized.
        For each attribute, max is the maximal value and min is the minimal. The normalization equation is: (data-min)/(max-min).
        :X: nd array object loaded from the file previously.
        :return: numpy array object containing the normalized data set
        """
        number_of_columns = data.shape[1]
        for i in range(1, number_of_columns - 1):
            v = data[:, i]
            maximum_value = v.max()
            minimum_value = v.min()
            denominator = maximum_value - minimum_value
            normalized_column = (v - minimum_value) / (denominator)
            data[:, i] = normalized_column

        return data

    def calculateCost(self, X, weights):
        """
        Function for calculating the convergence during the gradient descent
        :x: input data set.
        :weights: weights on which the error has to be calculated
        :return: return the error rate values
        """
        n_data_points, n_features = X.shape

        x = X[:, :-1]
        y = X[:, -1]
        yhat = x@weights
        cost = (1/(2*n_data_points))*np.sum((yhat - y)**2)
        return cost

    def stochasticGD(self, X):
        pass

    def gradient_descent(self, data, weights, learning_rate, n_iters):
        n_data_points, n_features = data.shape
        history = np.zeros((n_iters, 1))
        X = data[:, :-1]
        y = data[:, -1]
        for i in range(n_iters):
            weights = weights - ((learning_rate/n_data_points) *
                                 (X.T @ (X @ weights - y)))
            history[i] = self.calculateCost(data, weights)
        return (history, weights)

    def splitTT(self, X, percentTrain):
        pass

    def splitCV(self, X, folds):
        pass
