import numpy as np


class Logistic_Regression:

    def __init__(self, fileName):
        self.fileName = fileName
        self.tolerance = 1e-5

    def dataLoad(self, fileName):
        print(f'Loading file {fileName}')
        data = np.genfromtxt(
            self.fileName, delimiter=',')
        print(f'Loaded file {fileName} with {data.shape} records')
        return data

    def addZeroColumn(self, data):
        """
        AddZeroColumn adds a col 0 to the data set to handle the bias
        :X: nd array object loaded from the file previously.
        :return: numpy array object containing the new column
        """
        number_of_rows = data.shape[0]
        first_column = np.ones([number_of_rows])
        data = np.insert(data, 0, first_column, axis=1)
        print(f'Added a new X0 column {data.shape}')
        return data

    def dataNorm(self, data):
        """
        DataNorm normalizes all the columns in the data set except the last column as the output is not to be normalized.
        For each attribute, max is the maximal value and min is the minimal. The normalization equation is: (data-min)/(max-min).
        :X: nd array object loaded from the file previously.
        :return: numpy array object containing the normalized data set
        """
        data = self.addZeroColumn(data)
        number_of_columns = data.shape[1]
        for i in range(1, number_of_columns - 1):
            v = data[:, i]
            maximum_value = v.max()
            minimum_value = v.min()
            denominator = maximum_value - minimum_value
            normalized_column = (v - minimum_value) / (denominator)
            data[:, i] = normalized_column

        print(f'Normalized data with X0 column {data.shape}')

        return data

    def printMeanAndSum(self, data):
        """
        Print the mean and Sum of all the columns for validation purpose.
        :X: nd array object loaded from the file previously.
        """
        column_names = ["Column", "Attribute", "Mean", "Sum"]
        attribute_names = ['Col1', 'Variance',
                           'Skewness', 'Kurtosis', 'Entropy', 'Class']
        format_row = "{:^20}" * (len(column_names)+1)
        print(format_row.format("", *column_names))

        number_of_columns = data.shape[1]
        for i in range(number_of_columns):
            mean_value = np.mean(data[:, i], axis=0)
            sum_value = np.sum(data[:, i], axis=0)
            column_number = 'Col' + str(i+1)
            row = [column_number, attribute_names[i],  mean_value, sum_value]
            print(format_row.format('', *row))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def errCompute(self, X, weights):
        x = X[:, :-1]
        y = X[:, -1]
        n = X.shape[0]
        z = x@weights
        yhat = self.sigmoid(z)

        predict_1 = y * np.log(yhat)
        predict_0 = (1 - y) * np.log(1 - yhat)
        summation = np.mean(-(predict_1 + predict_0))
        return summation

    def fit(self, X, learning_rate, epoch):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        x = X[:, :-1]
        y = X[:, -1]

        # gradient descent
        for _ in range(epoch):
            z = x@self.weights
            yhat = self.sigmoid(z)
            dw = (1/n_samples) * np.dot(x.T, (yhat - y))
            self.weights -= learning_rate * dw

    def gradient_descent(self, X, learning_rate):
        n_samples, n_features = X.shape
        x = X[:, :-1]
        y = X[:, -1]

        z = x@self.weights
        yhat = self.sigmoid(z).reshape(n_samples)
        dw = (1/n_samples) * np.dot(x.T, (yhat - y))
        self.weights -= learning_rate * dw

    def predict(self, X):
        x = X[:, :-1]
        y = X[:, -1]
        z = x@self.weights
        y_prediction = self.sigmoid(z)
        y_prediction_cls = [1 if i > 0.5 else 0 for i in y_prediction]
        accuracy = self.accuracy(y, y_prediction_cls)
        return y_prediction_cls, accuracy

    def accuracy(self, y_true, y_prediction):
        accuracy = np.sum(y_true == y_prediction) / len(y_true)
        return accuracy

    def stochasticGD(self, X, weights, learning_rate, epoch):
        previous_loss = -float('inf')
        n_samples, n_features = X.shape
        self.weights = weights  # np.zeros(n_features-1)
        converged = False
        number_of_runs = 0
        for _ in range(epoch):

            loss = self.errCompute(X, weights)
            number_of_runs += 1
            # convergence check
            if abs(previous_loss - loss) < self.tolerance:
                converged = True
                break
            else:
                previous_loss = loss
            self.gradient_descent(X, learning_rate)
        print(f"Number of runs {number_of_runs}")
        return self.weights

    # def stochasticGD(self, X, weights, learning_rate, epoch):
    #     loss = []
    #     previous_loss = -float('inf')
    #     self.converged = False
    #     x = X[:, :-1]
    #     y = X[:, -1]
    #     n = X.shape[0]

    #     for iteration in range(epoch):
    #         z = x@weights
    #         yhat = self.sigmoid(z)
    #         print(yhat)
    #         weights -= learning_rate * np.dot(x.T,  (yhat - y))
    #         loss.append(self.errCompute(X, weights))
    #     self.weights = weights
    #     self.loss = loss

    #     return self.weights
