import numpy as np
import csv
import time
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# np.seterr(divide='ignore', invalid='ignore')


class Linear_Regression:
    def __init__(self, X, y, alpha=0.01, iterations=200000):

        self.train_percentages = [0.7]
        self.cross_fold_values = [15]
        self.learning_rates = [.001]
        self.statistics = []
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

    @classmethod
    def splitCV(cls, X, folds):
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

    def trainandTest(self, X_Train, X_Test, learning_rate):
        """
        Takes in the normalized dataset X_train,X_test and learning rate
        :X_train: train data set.
        :X_test: test data set.
        :learning_rate: Learning Rate.
        :return:  prediction values, weights, accuracy score
        """

        X = X_Train[:, :-1]
        Y = X_Train[:, -1]
        regressor = Linear_Regression(X, Y, alpha=learning_rate).model()
        train_accuracy = regressor.accuracy()
        # print(train_accuracy)

        test_X = X_Test[:, :-1]
        test_Y = X_Test[:, -1]

        test_accuracy_score = regressor.accuracy(test_X, test_Y)
        # print(test_accuracy_score)

        y_prediction_value = regressor.predict(X)
        weights = regressor.get_weights()

        return y_prediction_value, weights, test_accuracy_score

    def k_fold_cross_validation(self, X, folds, learning_rate):
        """
        Takes in the Normalized array and number of k-values needed and the number of folds needed.
        The function would iterate over all the fold partitions except the fold in enumeration
        and get the other folds and call the Linear regression algorithm those many times to get the accuracy.
        The returned accuracy is the mean of the individual fold accuracies and also a list of predicted labels
        which would be used in the Classification report
        :X: Train Data set which is a nd array object normalized.
        :k: k-value.
        :folds: number of folds for which knn needs to be done.
        :return: accuracy of this iteration and list of predicted outputs
        """
        weights_accuracy_vector = []
        accuracy_listing = []
        k_fold_partitions = self.splitCV(X, folds)
        for index, item in enumerate(k_fold_partitions):
            cross_validation_dataset = item
            list_of_items_from_zero_to_index = k_fold_partitions[0:index]
            list_of_items_from_index_to_end = k_fold_partitions[index+1:]
            total_train_list = list_of_items_from_zero_to_index + \
                list_of_items_from_index_to_end
            train_data_set = np.vstack(total_train_list)
            y_prediction_value, theta, accuracy_score = self.trainandTest(
                train_data_set, cross_validation_dataset, learning_rate)
            # print(f'Thetha is : {theta}')
            weights_accuracy_vector.append(
                (index, accuracy_score, theta, learning_rate, cross_validation_dataset, y_prediction_value))
            if accuracy_score is not np.nan:
                accuracy_listing.append(accuracy_score)
        print(f'Accuracy Listing {accuracy_listing}')
        accuracy_average = np.average(accuracy_listing)

        print(
            f'Folds : {folds}, Accuracy Average : {accuracy_average} ')
        return accuracy_average, weights_accuracy_vector

    def trigger_k_fold_cross_validation(self, X):
        """
        Trigger the train and given the input data set 
        :X: Train Data set which is a nd array object normalized.
        """
        for train_percentage in self.train_percentages:
            print('*'*20)
            print(f' Training with {train_percentage*100} % data')
            print('-'*10)
            x_train, x_test = Linear_Regression.splitTT(X, train_percentage)
            accuracy_listing = []
            for cross_fold in self.cross_fold_values:
                for learning_rate in self.learning_rates:
                    tic = time.perf_counter()
                    accuracy_cross_fold, weights_accuracy_vector = self.k_fold_cross_validation(
                        x_train, cross_fold, learning_rate)
                    toc = time.perf_counter()
                    time_taken = toc - tic
                    accuracy_listing.append(accuracy_cross_fold)
                    datapoint = ('Linear Regression', train_percentage *
                                 100, learning_rate, cross_fold, accuracy_cross_fold, time_taken, weights_accuracy_vector)
                    self.statistics.append((datapoint))

            average_accuracy_score = np.mean(accuracy_listing)
            print(
                f' Average accuracy for all cross folds : {average_accuracy_score:.3f}')
            print('Mean Accuracy: ' +
                  "{:.2f}".format(average_accuracy_score*100)+'%')
        print('Statistics')
        # print(self.statistics)
        return self.statistics


# https://towardsdatascience.com/linear-regression-from-scratch-with-numpy-implementation-finally-8e617d8e274c
