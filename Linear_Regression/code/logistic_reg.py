import numpy as np
import csv
import time
from sklearn.metrics import roc_curve
from matplotlib import pyplot


class Logistic_Regression:

    def __init__(self, fileName):
        self.fileName = fileName
        self.tolerance = 1e-5
        self.train_percentages = [0.6]
        self.cross_fold_values = [5]
        # self.learning_rates = [.01, .02, .03, 1, 2, 3, 4, 5]
        self.learning_rates = [.1, .2, .3, .4, 1]
        self.statistics = []

    def dataLoad(self, fileName):
        """
        dataLoad loads data from filename provided into a numpy array X and returns it.

        :filename filename: File name to be loaded to a numpy nd array object.
        :return: numpy array object containing the entire data set
        """
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
        # print(f'Added a new X0 column {data.shape}')
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

        # print(f'Normalized data with X0 column {data.shape}')

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
        """
        Calculate sigmoid on the WX values(x@weights).
        :z: predicted values.
        :return: sigmoid values
        """
        return 1 / (1 + np.exp(-z))

    def errCompute(self, X, weights):
        """
        Function for calculating the convergence during the gradient descent
        :x: input data set.
        :weights: weights on which the error has to be calculated
        :return: return the error rate values
        """
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
        """
        Function for fitting the data points with learning rate and epoch
        :x: input data set.
        :weights: weights
        :epoch: number of times the data is parsed.
        """
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
        """
        Function for gradient fitting weights on the input data set
        :x: input data set.
        :learning_rate: learning rate
        :return: setting the weights
        """
        n_samples, n_features = X.shape
        x = X[:, :-1]
        y = X[:, -1]

        z = x@self.weights
        yhat = self.sigmoid(z).reshape(n_samples)
        dw = (1/n_samples) * np.dot(x.T, (yhat - y))
        self.weights -= learning_rate * dw

    def predict(self, X, weight):
        """
        Function for predicting the y values given a set of weights
        :x: input data set.
        :weight: weights
        :return: predicted y values and accuracy value
        """
        weight = weight.reshape(len(weight))
        x = X[:, :-1]
        y = X[:, -1]
        z = x@weight  # self.weights
        y_prediction = self.sigmoid(z)
        y_prediction_cls = [1 if i > 0.5 else 0 for i in y_prediction]
        accuracy = self.accuracy(y, y_prediction_cls)
        return y_prediction_cls, accuracy

    def accuracy(self, y_true, y_prediction):
        """
        Function for calculating the accuracy given y true values and y prediction values
        :y_true: y true values.
        :y_prediction: y prediction values
        :return:accuracy values
        """
        accuracy = np.sum(y_true == y_prediction) / len(y_true)
        return accuracy

    def stochasticGD1(self, X, weights, learning_rate, epoch):
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
                # print(f'Within tolerace limit of {self.tolerance}')
                converged = True
                break
            else:
                previous_loss = loss
            self.gradient_descent(X, learning_rate)
        print(f"Number of runs {number_of_runs}")
        return self.weights.reshape((len(self.weights), 1))

    def stochasticGD(self, X, weights, learning_rate, max_iter, batch_size=300):
        """
        This function takes in dataset X_norm (should be shuffled), theta , learning rate alpha , 
        and maximal iterations num_iters. It returns the learned theta.
        :X: input feature values.
        :learning_rate: learning rate alphas value
        :max_iter: maximum number of iterations which would accomodate the different batch sizes
        :batch_size: batch sizes after which the weights are re-calibrated
        :return: weights after doing the descent
        """

        weights = weights.reshape(len(weights))
        self.weights = weights
        previous_loss = -float('inf')
        iterations = 0
        folds = batch_size
        error_rates = []

        for _ in range(max_iter):

            k_fold_partitions = self.splitCV(X, folds)
            for index, item in enumerate(k_fold_partitions):
                self.gradient_descent(item, learning_rate)
            loss = self.errCompute(X, weights)
            error_rates.append(loss)
            if abs(previous_loss - loss) < self.tolerance:
                # print(f'Within tolerace limit of {self.tolerance}')
                break
            else:
                previous_loss = loss
            iterations += 1
        print(f"Number of runs {iterations}")
        self.iterations = iterations
        self.error_rates = error_rates
        return self.weights.reshape((len(self.weights), 1))

    def stochasticGDMiniBatch(self, X, weights, learning_rate, max_iter, batch_size):

        weights = weights.reshape(len(weights))
        self.weights = weights
        previous_loss = -float('inf')
        iterations = 0
        folds = batch_size
        error_rates = []

        for _ in range(max_iter):

            k_fold_partitions = self.splitCV(X, folds)
            for index, item in enumerate(k_fold_partitions):
                self.gradient_descent(item, learning_rate)
            loss = self.errCompute(X, weights)
            error_rates.append(loss)
            if abs(previous_loss - loss) < self.tolerance:
                # print(f'Within tolerace limit of {self.tolerance}')
                break
            else:
                previous_loss = loss
            iterations += 1
        print(f"Number of runs {iterations}")
        self.iterations = iterations
        self.error_rates = error_rates
        return self.weights.reshape((len(self.weights), 1))

    def trainandTest(self, X_Train, X_Test, learning_rate):
        """
        Takes in the normalized dataset X_train,X_test and learning rate
        :X_train: train data set.
        :X_test: test data set.
        :learning_rate: Learning Rate.
        :return: accuracy, prediction classes, weights, epochs and error rates
        """

        classifier = Logistic_Regression("")
        data = classifier.dataNorm(X_Train)
        test_data = classifier.dataNorm(X_Test)
        theta = np.zeros((data.shape[1]-1, 1))
        theta = classifier.stochasticGD(
            data, theta, learning_rate, len(X_Train)*20)
        epoch = classifier.iterations
        error_rates = classifier.error_rates

        y_prediction_cls, accuracy = classifier.predict(test_data, theta)
        return accuracy, y_prediction_cls, theta, epoch, error_rates

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

    def k_fold_cross_validation(self, X, folds, learning_rate):
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
        weights_accuracy_vector = []
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
            accuracy_for_cross_validation, actual_predicted_labels_from_partition, theta, epochs, error_rates = self.trainandTest(
                train_data_set, cross_validation_dataset, learning_rate)
            print(f'Thetha is : {theta}')
            # print(f'Error rate length : {len(error_rates)}')
            # print(f'Error Rates : {error_rates}')
            weights_accuracy_vector.append(
                (index, accuracy_for_cross_validation, theta, epochs, error_rates))
            accuracy_listing.append(accuracy_for_cross_validation)
            actual_predicted_labels.append(
                actual_predicted_labels_from_partition)
        print(f'Accuracy Listing {accuracy_listing}')
        accuracy_average = np.average(accuracy_listing)

        print(
            f'Folds : {folds}, Accuracy Average : {accuracy_average} ')
        return accuracy_average, actual_predicted_labels, weights_accuracy_vector

    def trigger_k_fold_cross_validation(self, X):
        """
        Trigger the train and given the input data set 
        :X: Train Data set which is a nd array object normalized.
        """
        for train_percentage in self.train_percentages:
            print('*'*20)
            print(f' Training with {train_percentage*100} % data')
            print('-'*10)
            x_train, x_test = self.splitTT(X, train_percentage)
            accuracy_listing = []
            for cross_fold in self.cross_fold_values:
                for learning_rate in self.learning_rates:
                    tic = time.perf_counter()
                    accuracy_cross_fold, actual_predicted_labels, weights_accuracy_vector = self.k_fold_cross_validation(
                        x_train, cross_fold, learning_rate)
                    toc = time.perf_counter()
                    time_taken = toc - tic
                    accuracy_listing.append(accuracy_cross_fold)
                    datapoint = ('Logistic Regression', train_percentage *
                                 100, learning_rate, cross_fold, accuracy_cross_fold, time_taken, weights_accuracy_vector)
                    self.statistics.append((datapoint))

            average_accuracy_score = np.mean(accuracy_listing)
            print(
                f' Average accuracy for all cross folds : {average_accuracy_score:.3f}')
            print('Mean Accuracy: ' +
                  "{:.2f}".format(average_accuracy_score*100)+'%')
        print('Statistics')
        # print(self.statistics)
        self.print_statistics(self.statistics)
        self.print_Error_Rate(self.statistics)
        selected_weights = self.get_weights_for_final_testing(
            self.statistics, 1)
        # self.draw_roc_curve(x_test, selected_weights)

    def draw_roc_curve(self, test_data, theta):
        classifier = Logistic_Regression("")
        y_prediction_cls, accuracy = classifier.predict(test_data, theta)
        y = test_data[:, -1]
        fpr, tpr, _ = roc_curve(y, y_prediction_cls)
        pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        pyplot.plot(fpr, tpr, marker='.', label='Logistic')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
        pass

    def print_Error_Rate(self, statistic):
        with open('output/error_rate.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["learning_Rate", "Cross_Validation_Fold", "Epochs", "index", "Error_Rate"])
            for index, stat in enumerate(statistic):
                for index2, weight_vector in enumerate(stat[6]):
                    epochs = weight_vector[3]
                    error_rate = weight_vector[4]
                    cross_fold = 'Set - '+str(weight_vector[0])
                    learning_Rate = stat[2]
                    for index, error in enumerate(error_rate):
                        writer.writerow(
                            [learning_Rate, cross_fold, epochs, index, error])

    def get_weights_for_final_testing(self, statistics, learning_rate):
        for index, stat in enumerate(statistics):
            for index2, weight_vector in enumerate(stat[6]):
                weights_array_vector = weight_vector[2].reshape(
                    len(weight_vector[2]))

                cross_fold = weight_vector[0]
                lrate = stat[2]

                if cross_fold == 0 and lrate == learning_rate:
                    choosen_weight = weight_vector[2]
                    return choosen_weight

    def print_statistics(self, statistics):
        """
        Print the statistics to be used for subsequent analysis in jupyter notebook for drawing charts.
        :statistics: nd array of tuples.
        """
        with open('output/statistics.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Method", "Train_Percentage",
                             "learning_Rate", "Cross_Validation_Fold", "Accuracy", "Epochs", "Bias", "Variance", "Skewness", "Kurtosis", "Entropy"])
            for index, stat in enumerate(statistics):
                for index2, weight_vector in enumerate(stat[6]):
                    weights_array_vector = weight_vector[2].reshape(
                        len(weight_vector[2]))
                    accuracy = weight_vector[1]
                    epochs = weight_vector[3]
                    error_rate = weight_vector[4]
                    print(error_rate)
                    cross_fold = weight_vector[0]
                    bias = weights_array_vector[0]
                    Variance = weights_array_vector[1]
                    Skewness = weights_array_vector[2]
                    Kurtosis = weights_array_vector[3]
                    Entropy = weights_array_vector[4]
                    writer.writerow([stat[0], stat[1],
                                     stat[2], 'Set - '+str(cross_fold),  accuracy, epochs, bias, Variance, Skewness, Kurtosis, Entropy])
