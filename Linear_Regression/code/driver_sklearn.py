from sklearn.datasets import load_boston
from Linear_Regression import Linear_Regression as classifier
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

data_file_name = 'data/housing.data'


data = data = np.genfromtxt(
    data_file_name)
# data = classifier.dataStandardization(data)

data = np.delete(data, [3, 1], 1)

print(data[0:])
x_train, x_test = classifier.splitTT(data, .8)

print(x_train.shape)
print(x_test.shape)

X = x_train[:, :-1]
Y = x_train[:, -1]

regressor = classifier(X, Y).model()
our_train_accuracy, rmse = regressor.accuracy()

print(our_train_accuracy)

X = x_test[:, :-1]
Y = x_test[:, -1]
our_test_accuracy, rmse = regressor.accuracy(X, Y)
print(our_test_accuracy)

# y_pred = our_regressor.predict(X)
# # print(y_pred)
# accuracy = our_regressor.accuracy(Y, y_pred)
# print(accuracy)


# X = x_test[:, :-1]
# Y = x_test[:, -1]


# our_test_accuracy = our_regressor.score(X_test, y_test)


statistics = classifier(X, Y).trigger_k_fold_cross_validation(data)
