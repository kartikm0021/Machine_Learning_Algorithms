import numpy as np


class Logistic_Regression:

    def __init__(self, fileName):
        self.fileName = fileName
        self.tolerance = 1e-5
        self.train_percentages = [1, 0.7, 0.6, 0.5]
        self.cross_fold_values = [5, 10, 15]

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
        weights = weights.reshape(len(weights))
        x = X[:, :-1]
        y = X[:, -1]

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

    def predict(self, X, weight):
        weight = weight.reshape(len(weight))
        x = X[:, :-1]
        y = X[:, -1]
        z = x@weight  # self.weights
        y_prediction = self.sigmoid(z)
        y_prediction_cls = [1 if i > 0.5 else 0 for i in y_prediction]
        accuracy = self.accuracy(y, y_prediction_cls)
        return y_prediction_cls, accuracy

    def accuracy(self, y_true, y_prediction):
        accuracy = np.sum(y_true == y_prediction) / len(y_true)
        return accuracy

    def stochasticGD(self, X, weights, learning_rate, epoch):
        weights = weights.reshape(len(weights))
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
                print(f'Within tolerace limit of {self.tolerance}')
                converged = True
                break
            else:
                previous_loss = loss
            self.gradient_descent(X, learning_rate)
        print(f"Number of runs {number_of_runs}")
        return self.weights.reshape((len(self.weights), 1))

    def splitTT(self, X, percentTrain):
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

    def splitCV(self, X, folds):
        """
        Takes in the normalized dataset X_norm ,and the number of folds needed
        This would split the number of records equlilantly in every partition. If k is a number which cannot be equliably splitted
        it would distribute the extra records into all the partitions.
        :X: nd array object normalized.
        :folds: number of folds needed.
        :return: list of numpy array objects containing the different folds or partitions
        """
        np.random.shuffle(X)
        split_array = np.array_split(X, folds)
        return split_array

    def k_fold_cross_validation(self, X, k, folds):
        """
        Takes in the Normalized array and number of k-values needed and the number of folds needed.
        The function would iterate over all the fold partitions except the fold in enumeration 
        and get the other folds and call the knn algorithm those many times to get the accuracy. 
        The returned accuracy is the mean of the individual fold accuracies and also a list of predicted labels
        which would be used in the Classification report
        :X: Train Data set which is a nd array object normalized.
        :k: k-value. 
        :folds: number of folds for which knn needs to be done.
        :return: accuracy of this iteration and list of predicted outputs
        """
        accuracy_listing = []
        actual_predicted_labels = []
        k_fold_partitions = self.splitCV(X, folds)
        for index, item in enumerate(k_fold_partitions):
            cross_validation_dataset = item
            list_of_items_from_zero_to_index = k_fold_partitions[0:index]
            list_of_items_from_index_to_end = k_fold_partitions[index+1:]
            total_train_list = list_of_items_from_zero_to_index + \
                list_of_items_from_index_to_end
            train_data_set = np.vstack(total_train_list)
            accuracy_for_cross_validation, actual_predicted_labels_from_partition = knn(
                train_data_set, cross_validation_dataset, k)
            accuracy_listing.append(accuracy_for_cross_validation)
            actual_predicted_labels.append(
                actual_predicted_labels_from_partition)

        accuracy_average = np.average(accuracy_listing)

        print(
            f'k-value :{k}, Folds : {folds}, Accuracy Average : {accuracy_average} ')
        return accuracy_average, actual_predicted_labels
