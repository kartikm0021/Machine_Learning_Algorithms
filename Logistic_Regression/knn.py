import numpy as np
import csv
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def loadData(filename):
    """
    loadData loads data from filename provided into a numpy array X and returns it.

    :filename filename: File name to be loaded to a numpy nd array object.
    :return: numpy array object containing the entire data set
    """
    X = []
    count = 0

    text_file = open(filename, "r")
    lines = text_file.readlines()

    for line in lines:
        X.append([])
        words = line.split(",")
        # convert value of first attribute into float
        for word in words:
            if (word == 'M'):
                word = 0.333
            if (word == 'F'):
                word = 0.666
            if (word == 'I'):
                word = 1
            X[count].append(float(word))
        count += 1

    return np.asarray(X)


def dataNorm(X):
    """
    DataNorm normalizes all the columns in the data set except the last column as the output is not to be normalized.
    For each attribute, max is the maximal value and min is the minimal. The normalization equation is: (data-min)/(max-min).
    :X: nd array object loaded from the file previously.
    :return: numpy array object containing the normalized data set
    """
    number_of_columns = X.shape[1]
    for i in range(number_of_columns - 1):
        v = X[:, i]
        maximum_value = v.max()
        minimum_value = v.min()
        denominator = maximum_value - minimum_value
        normalized_column = (v - minimum_value) / (denominator)
        X[:, i] = normalized_column

    return X


def printMeanAndSum(X):
    """
    Print the mean and Sum of all the columns for validation purpose.
    :X: nd array object loaded from the file previously.
    """
    print('Printing the Normalized data set mean and sum')
    column_names = ["Column", "Attribute", "Mean", "Sum"]
    attribute_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                       'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings(output)']
    format_row = "{:^20}" * (len(column_names)+1)
    print(format_row.format("", *column_names))

    number_of_columns = X.shape[1]
    for i in range(number_of_columns):
        mean_value = np.mean(X[:, i], axis=0)
        sum_value = np.sum(X[:, i], axis=0)
        column_number = 'Col' + str(i+1)
        row = [column_number, attribute_names[i],  mean_value, sum_value]
        print(format_row.format('', *row))


def splitTT(X, percentTrain):
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


def splitCV(X, folds):
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


def knn(X_Train, X_Test, k):
    """
    Takes in the X_Train and X_Test ,and the number of k-values needed
    This would figure out all the nearest neighbours with respect to the test set 
    and the training set and return the accuracy
    it would distribute the extra records into all the partitions.
    :X_Train: Train Data set which is a nd array object normalized.
    :X_Test: Test data set which is an nd array object normalized
    :k: k-value.
    :return: accuracy of this iteration and list of predicted labels to be used in classification report
    """

    test_excluding_output_column = X_Test[:, :8]
    train_excluding_output_column = X_Train[:, :8]
    dists = (-2 * np.dot(test_excluding_output_column, train_excluding_output_column.T) + np.sum(train_excluding_output_column**2, axis=1) +
             np.sum(test_excluding_output_column**2, axis=1)[:, np.newaxis])**0.5

    correct_count = 0
    actual_predicted_labels = []
    sorted_distances = np.argsort(dists)
    k_nearest_distances_indexes = sorted_distances[:, :k]

    for item_index, neighbour_index_listing in enumerate(k_nearest_distances_indexes):
        nearest_neighbours = []
        actual_label = X_Test[item_index][8]
        counter = {}
        for neighbour_index in neighbour_index_listing:
            class_label_in_train_data_set = X_Train[neighbour_index][-1]
            distance_from_this_point = dists[item_index][neighbour_index]
            weight = (1/distance_from_this_point)
            nearest_neighbours.append(
                (class_label_in_train_data_set, distance_from_this_point, weight))
            counter[class_label_in_train_data_set] = counter.get(
                class_label_in_train_data_set, 0)+weight
        predicted_label = max(counter, key=counter.get)
        actual_predicted_labels.append((actual_label, predicted_label))
        if predicted_label == actual_label:
            correct_count += 1

    accuracy = (correct_count / len(X_Test)) * 100

    return accuracy, actual_predicted_labels


def k_fold_cross_validation(X, k, folds):
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
    k_fold_partitions = splitCV(X, folds)
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
        actual_predicted_labels.append(actual_predicted_labels_from_partition)

    accuracy_average = np.average(accuracy_listing)

    print(
        f'k-value :{k}, Folds : {folds}, Accuracy Average : {accuracy_average} ')
    return accuracy_average, actual_predicted_labels


def print_statistics(statistics):
    """
    Print the statistics to be used for subsequent analysis in jupyter notebook for drawing charts.
    :statistics: nd array of tuples.
    """
    with open('statistics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNo", "Method", "Train_Percentage",
                         "K_value", "Cross_Validation_Fold", "Accuracy", "Run_Time"])
        for index, stat in enumerate(statistics):
            writer.writerow([index+1, stat[0], stat[1],
                             stat[2], stat[3], stat[4], stat[5]])


def print_classification_report(actual_predicted_labels, unique_labels):
    """
    Print the classification report which is one of the output reports needed
    :actual_predicted_labels: actual predicted labels.
    :unique_labels: unique labels.
    """
    # print(f' Actual Predicted Labels {len(actual_predicted_labels)}')
    actual_prediction_values = [
        item for items in actual_predicted_labels for item in items]
    # print(f'Length of tuple list is {len(actual_prediction_values)}')

    actual_prediction_values_list = list(
        map(list, zip(*actual_prediction_values)))
    actual_values = actual_prediction_values_list[0]
    predicted_values = actual_prediction_values_list[1]
    count_of_missing = set(actual_values) - set(predicted_values)
    # print('Printing frequencies of actual values')
    # print_frequencies(actual_values)
    # print(f'count_of_missing : {count_of_missing}')
    report = classification_report(actual_values,
                                   predicted_values)
    print(report)


def print_frequencies(listing):
    """
    Helper needed to get the list of frequencies of records from a list
    :listing: list of records.
    """
    freq_list = []
    a_l = list(set(listing))

    for x in a_l:
        freq_list.append(listing.count(x))

    # print('Freq', freq_list)
    # print('number', a_l)


def knnMain_New(filename):
    """
    Driver method to get the filename, normalize the records,
    Split into Train and Test records
    Split using k-fold cross validation
    Call KNN() method for both the above splits.
    Generate output and store it in a file for further analysis
    Generate the classification report

    :filename: file name.
    """

    k_values = [1, 5, 10, 15, 20]
    train_percentages = [1, 0.7, 0.6, 0.5]
    cross_fold_values = [5, 10, 15]

    X = loadData(filename)
    X_norm = dataNorm(X)
    printMeanAndSum(X_norm)

    statistics = []
    number_of_columns = X.shape[1]
    unique_labels = np.unique(X_norm[:, (number_of_columns - 1)])
    # print(f'Unique Labels {unique_labels}')

    for train_percentage in train_percentages[0:1]:
        print('*'*20)
        print(f' Training with {train_percentage*100} % data')
        print('-'*10)
        x_train, x_test = splitTT(X_norm, train_percentage)
        for k_value in k_values:
            # print(f'K-Value : {k_value}')
            accuracy_listing = []
            for cross_fold in cross_fold_values:
                # print(f'Cross Fold : {cross_fold}')
                tic = time.perf_counter()
                accuracy_cross_fold, actual_predicted_labels = k_fold_cross_validation(
                    x_train, k_value, cross_fold)
                toc = time.perf_counter()
                time_taken = toc - tic
                # print(
                #     f"Time taken to run Cross Validation : {time_taken:0.4f} seconds")
                accuracy_listing.append(accuracy_cross_fold)
                datapoint = ('K-Fold-Cross-Validation', train_percentage*100, k_value,
                             cross_fold, accuracy_cross_fold, time_taken)
                statistics.append((datapoint))

                if k_value == 15 and cross_fold == 5:
                    # print('Printing frequencies of X_Train')
                    print_frequencies(
                        list(x_train[:, (number_of_columns - 1)]))
                    print_classification_report(
                        actual_predicted_labels, unique_labels)

            average_accuracy_score = np.mean(accuracy_listing)
            print(
                f' Average accuracy for all cross folds : {average_accuracy_score}')

    cross_fold = 0
    for train_percentage in train_percentages[1:]:
        x_train, x_test = splitTT(X_norm, train_percentage)
        for k_value in k_values:
            tic = time.perf_counter()
            accuracy, actual_predicted_labels = knn(x_train, x_test, k_value)
            toc = time.perf_counter()
            time_taken = toc - tic
            datapoint = ('Train and Test', train_percentage*100, k_value,
                         cross_fold, accuracy, time_taken)
            statistics.append(datapoint)

    print_statistics(statistics)


knnMain_New('abalone.data')
